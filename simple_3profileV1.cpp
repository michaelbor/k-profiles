/*  
 * Copyright (c) 2009 Carnegie Mellon University. 
 *     All rights reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing,
 *  software distributed under the License is distributed on an "AS
 *  IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 *  express or implied.  See the License for the specific language
 *  governing permissions and limitations under the License.
 *
 * For more about this software visit:
 *
 *      http://www.graphlab.ml.cmu.edu
 *
 */


#include <boost/unordered_set.hpp>
#include <graphlab.hpp>
#include <graphlab/ui/metrics_server.hpp>
#include <graphlab/macros_def.hpp>
#include <boost/array.hpp>
// #include <boost/vector.hpp>
/**
 *  
 * In this program we implement the "hash-table" version of the
 * "edge-iterator" algorithm described in
 * 
 *    T. Schank. Algorithmic Aspects of Triangle-Based Network Analysis.
 *    Phd in computer science, University Karlsruhe, 2007.
 *
 * The procedure is quite straightforward:
 *   - each vertex maintains a list of all of its neighbors in a hash table.
 *   - For each edge (u,v) in the graph, count the number of intersections
 *     of the neighbor set on u and the neighbor set on v.
 *   - We store the size of the intersection on the edge.
 * 
 * This will count every triangle exactly 3 times. Summing across all the
 * edges and dividing by 3 gives the desired result.
 *
 * The preprocessing stage take O(|E|) time, and it has been shown that this
 * algorithm takes $O(|E|^(3/2))$ time.
 *
 * If we only require total counts, we can introduce a optimization that is
 * similar to the "forward" algorithm
 * described in thesis above. Instead of maintaining a complete list of all
 * neighbors, each vertex only maintains a list of all neighbors with
 * ID greater than itself. This implicitly generates a topological sort
 * of the graph.
 *
 * Then you can see that each triangle
 *
 * \verbatim
  
     A----->C
     |     ^
     |   /
     v /
     B
   
 * \endverbatim
 * Must be counted only once. (Only when processing edge AB, can one
 * observe that A and B have intersecting out-neighbor sets).
 *
 *
 * \note The implementation here is built to be easy to understand
 * and not necessarily optimal. In particular the unordered_set is slow
 * for small number of entries. There is a much more efficient
 * (and substantially more complicated) version in undirected_triangle_count.cpp
 */

/*
 * Each vertex maintains a list of all its neighbors.
 * and a final count for the number of triangles it is involved in
 */
struct vertex_data_type {
  //this constructor ruins the global count!
  //vertex_data_type():num_triangles(0), num_wedges(0), num_disc(0), num_empty(0) { }
  //removing constructor also ruins global count, the line below only ruins global triangle count
  vertex_data_type():num_wedges(0) { }
  // A list of all its neighbors
  boost::unordered_set<graphlab::vertex_id_type> vid_set;
  // The number of triangles this vertex is involved it.
  // only used if "per vertex counting" is used
  size_t num_triangles;
  size_t num_wedges;
  size_t num_disc;
  size_t num_empty;
  
  vertex_data_type& operator+=(const vertex_data_type& other) {
    num_triangles += other.num_triangles;
    num_wedges += other.num_wedges;
    num_disc += other.num_disc;
	num_empty += other.num_empty;
    return *this;
  }
  //add to these??
  void save(graphlab::oarchive &oarc) const {
    //oarc << vid_set << num_triangles;
    oarc << vid_set << num_triangles << num_wedges << num_disc << num_empty;
  }
  void load(graphlab::iarchive &iarc) {
    iarc >> vid_set >> num_triangles >> num_wedges >> num_disc >>num_empty;
  }
};


/*
 * Each edge is simply a counter of triangles
 */
//typedef size_t edge_data_type;

struct edge_data_type {
  size_t n3;
  size_t n2;
  size_t n1;
  void save(graphlab::oarchive &oarc) const {
    //oarc << vid_set << num_triangles;
    oarc << n1 << n2 << n3;
  }
  void load(graphlab::iarchive &iarc) {
    iarc >> n1 >> n2 >> n3;
  }
};

// struct edge_data_type2 {
//   //will gather and reduce be easier with a boost vector? boost array?
//   boost::vector<size_t> n(3,0);
//   void save(graphlab::oarchive &oarc) const {
//     //oarc << vid_set << num_triangles;
//     oarc << n;
//   }
//   void load(graphlab::iarchive &iarc) {
//     iarc >> n;
//   }
// };

// typedef boost::std::vector<<size_t> edge_data_type(3);

struct set_edge_sum_gather {
  size_t n3;
  size_t n2;
  size_t n1;
  set_edge_sum_gather& operator+=(const set_edge_sum_gather& other) {
    n3 += other.n3;
    n2 += other.n2;
    n1 += other.n1;
    return *this;
  }

  // serialize
  void save(graphlab::oarchive& oarc) const {
    oarc << n1 << n2 << n3;
  }

  // deserialize
  void load(graphlab::iarchive& iarc) {
    iarc >> n1 >> n2 >> n3;
  }
};

// To collect the set of neighbors, we need a message type which is
// basically a set of vertex IDs

bool PER_VERTEX_COUNT = false;
size_t NUM_VERTICES = 0;


/*
 * This is the gathering type which accumulates an (unordered) set of
 * all neighboring vertices.
 * It is a simple wrapper around a boost::unordered_set with
 * an operator+= which simply performs a set union.
 *
 * This struct can be significantly accelerated for small sets.
 * Small collections of vertex IDs should not require the overhead
 * of the unordered_set.
 */
struct set_union_gather {
  boost::unordered_set<graphlab::vertex_id_type> vid_set;

  /*
   * Combining with another collection of vertices.
   * Union it into the current set.
   */
  set_union_gather& operator+=(const set_union_gather& other) {
    foreach(graphlab::vertex_id_type othervid, other.vid_set) {
      vid_set.insert(othervid);
    }
    return *this;
  }
  
  // serialize
  void save(graphlab::oarchive& oarc) const {
    oarc << vid_set;
  }

  // deserialize
  void load(graphlab::iarchive& iarc) {
    iarc >> vid_set;
  }
};

/*
 * Define the type of the graph
 */
typedef graphlab::distributed_graph<vertex_data_type,
                                    edge_data_type> graph_type;


/*
 * This class implements the triangle counting algorithm as described in
 * the header. On gather, we accumulate a set of all adjacent vertices.
 * If per_vertex output is not necessary, we can use the optimization
 * where each vertex only accumulates neighbors with greater vertex IDs.
 */
class triangle_count :
      public graphlab::ivertex_program<graph_type,
                                      set_union_gather>,
      /* I have no data. Just force it to POD */
      public graphlab::IS_POD_TYPE  {
public:
  // Gather on all edges
  edge_dir_type gather_edges(icontext_type& context,
                             const vertex_type& vertex) const {
    //sample edges here eventually, maybe by random edge.id() so consistent between the 2 engines?
    return graphlab::ALL_EDGES;
  } 

  /*
   * For each edge, figure out the ID of the "other" vertex
   * and accumulate a set of the neighborhood vertex IDs.
   */
  gather_type gather(icontext_type& context,
                     const vertex_type& vertex,
                     edge_type& edge) const {
    set_union_gather gather;
    // Insert the opposite end of the edge IF the opposite end has
    // ID greater than the current vertex
    // If we are getting per vertex counts, we need the entire neighborhood
    //only entire neighborhood
    vertex_id_type otherid = edge.source().id() == vertex.id() ?
                             edge.target().id() : edge.source().id();
    // if (PER_VERTEX_COUNT ||
    //     otherid > vertex.id()) gather.vid_set.insert(otherid);
    gather.vid_set.insert(otherid);
    return gather;
  }

  /*
   * the gather result now contains the vertex IDs in the neighborhood.
   * store it on the vertex. 
   */
  void apply(icontext_type& context, vertex_type& vertex,
             const gather_type& neighborhood) {
    vertex.data().vid_set = neighborhood.vid_set;
  } // end of apply

  /*
   * Scatter over all edges to compute the intersection.
   * I only need to touch each edge once, so if I scatter just on the
   * out edges, that is sufficient.
   */
  edge_dir_type scatter_edges(icontext_type& context,
                              const vertex_type& vertex) const {
    return graphlab::OUT_EDGES;
  }


  /*
   * Computes the size of the intersection of two unordered sets
   */
  static size_t count_set_intersect(
               const boost::unordered_set<vertex_id_type>& smaller_set,
               const boost::unordered_set<vertex_id_type>& larger_set) {
    size_t count = 0;
    foreach(vertex_id_type vid, smaller_set) {
      count += larger_set.count(vid);
    }
    return count;
  }

  /*
   * For each edge, count the intersection of the neighborhood of the
   * adjacent vertices. This is the number of triangles this edge is involved
   * in.
   */
  void scatter(icontext_type& context,
              const vertex_type& vertex,
              edge_type& edge) const {
    const vertex_data_type& srclist = edge.source().data();
    const vertex_data_type& targetlist = edge.target().data();
    int tmp= 0, tmp2 = 0;
    if (srclist.vid_set.size() >= targetlist.vid_set.size()) {
      tmp = count_set_intersect(targetlist.vid_set, srclist.vid_set);
    }
    else {
      //edge.data().int_card = count_set_intersect(srclist.vid_set, targetlist.vid_set);
      tmp = count_set_intersect(srclist.vid_set, targetlist.vid_set);
    }
    tmp2 = srclist.vid_set.size() + targetlist.vid_set.size();
    edge.data().n3 = tmp;
    edge.data().n2 =  tmp2 - 2*tmp;
    edge.data().n1 = NUM_VERTICES - (tmp2 - tmp);
  }
};



/*
 * This class is used in a second engine call if per vertex counts are needed.
 * The number of triangles a vertex is involved in can be computed easily
 * by summing over the number of triangles each adjacent edge is involved in
 * and dividing by 2. 
 */
class get_per_vertex_count :
      //public graphlab::ivertex_program<graph_type, size_t>,
      public graphlab::ivertex_program<graph_type, set_edge_sum_gather>,
      /* I have no data. Just force it to POD */
      public graphlab::IS_POD_TYPE  {
public:
  // Gather on all edges
  edge_dir_type gather_edges(icontext_type& context,
                             const vertex_type& vertex) const {
    return graphlab::ALL_EDGES;
  }
  // We gather the number of triangles each edge is involved in
  gather_type gather(icontext_type& context,
  //size_t gather(icontext_type& context,
                     const vertex_type& vertex,
                     edge_type& edge) const {
    //return edge.data();
    //more complex now
    set_edge_sum_gather gather;
    gather.n1 = edge.data().n1;
    gather.n2 = edge.data().n2;
    gather.n3 = edge.data().n3;
    return gather;
  }

  // boost:vector gather(icontext_type& context,
  // //size_t gather(icontext_type& context,
  //                    const vertex_type& vertex,
  //                    edge_type& edge) const {
  //   //return edge.data();
  //   //more complex now
  //   set_edge_sum_gather gather;
  //   gather.n1 = edge.data().n1;
  //   gather.n2 = edge.data().n2;
  //   gather.n3 = edge.data().n3;
  //   return gather;
  // }

  /* the gather result is the total sum of the number of triangles
   * each adjacent edge is involved in . Dividing by 2 gives the
   * desired result.
   */
  void apply(icontext_type& context, vertex_type& vertex,
             const gather_type& ecounts) { //edge_data_type??
    //no longer a gether_type& ??
    vertex.data().num_triangles = ecounts.n3 / 2;
    vertex.data().num_wedges = ecounts.n2 - pow(vertex.data().vid_set.size(),2)/2 + 
        vertex.data().num_triangles;
    vertex.data().num_disc = ecounts.n1;
    //how do I get graph.num_vertices() here?? NUM_VERTICES?
    vertex.data().num_empty = (NUM_VERTICES  - 1)*(NUM_VERTICES - 2)/2 - 
        (vertex.data().num_triangles + vertex.data().num_wedges + vertex.data().num_disc);
    //vertex.data().num_empty = (NUM_VERTICES  - 1)*(NUM_VERTICES - 2)/2 - 
    //    (ecounts.n1 + ecounts.n2 + ecounts.n3);
  }

  // No scatter
  edge_dir_type scatter_edges(icontext_type& context,
                             const vertex_type& vertex) const {
    return graphlab::NO_EDGES;
  }


};


/* Used to sum over all the edges in the graph in a
 * map_reduce_edges call
 * to get the total number of triangles
 */
////fixme...
// size_t get_edge_data(const graph_type::edge_type& e) {
//   return e.data();
// }

vertex_data_type get_vertex_data(const graph_type::vertex_type& v) {
  //denom = (NUM_VERTICES)*(NUM_VERTICES-1)*(NUM_VERTICES-2)(NUM_VERTICES-3)/24.; //divide by |V| choose 4, do this here?
  //denom = 1;
  // vertex_data_type finalout;
  // finalout.data().num_triangles = v.data().num_triangles/(3*denom);
  // finalout.data().num_wedges = v.data().num_wedges/(3*denom);
  // finalout.data().num_disc = v.data().num_disc/(2*denom);
  // finalout.data().num_empty = v.data().num_empty/(3*denom);
  return v.data();
}


/*
 * A saver which saves a file where each line is a vid / # triangles pair
 */
struct save_triangle_count{
  std::string save_vertex(graph_type::vertex_type v) { 
    return graphlab::tostr(v.id()) + "\t" +
           graphlab::tostr(v.data().num_triangles) + "\t" +
           graphlab::tostr(v.data().num_wedges) + "\t" +
           graphlab::tostr(v.data().num_disc) + "\t" +
           graphlab::tostr(v.data().num_empty) + "\n";
  }
  std::string save_edge(graph_type::edge_type e) {
    return "";
  }
};


int main(int argc, char** argv) {
  std::cout << "This program counts the exact number of triangles in the "
            "provided graph.\n\n";

  graphlab::command_line_options clopts("Exact Triangle Counting. "
    "Given a graph, this program computes the total number of triangles "
    "in the graph. An option (per_vertex) is also provided which "
    "computes for each vertex, the number of triangles it is involved in."
    "The algorithm assumes that each undirected edge appears exactly once "
    "in the graph input. If edges may appear more than once, this procedure "
    "will over count.");
  std::string prefix, format;
  std::string per_vertex;
  clopts.attach_option("graph", prefix,
                       "Graph input. reads all graphs matching prefix*");
  clopts.attach_option("format", format,
                       "The graph format");
  clopts.attach_option("per_vertex", per_vertex,
                       "If not empty, will count the number of "
                       "triangles each vertex belongs to and "
                       "save to file with prefix \"[per_vertex]\". "
                       "The algorithm used is slightly different "
                       "and thus will be a little slower");
  
  if(!clopts.parse(argc, argv)) return EXIT_FAILURE;
  if (prefix == "") {
    std::cout << "--graph is not optional\n";
    clopts.print_description();
    return EXIT_FAILURE;
  }
  else if (format == "") {
    std::cout << "--format is not optional\n";
    clopts.print_description();
    return EXIT_FAILURE;
  }


  if (per_vertex != "") PER_VERTEX_COUNT = true;
  // Initialize control plane using mpi
  graphlab::mpi_tools::init(argc, argv);
  graphlab::distributed_control dc;

  graphlab::launch_metric_server();
  // load graph
  graph_type graph(dc, clopts);
  graph.load_format(prefix, format);
  graph.finalize();
  dc.cout() << "Number of vertices: " << graph.num_vertices() << std::endl
            << "Number of edges:    " << graph.num_edges() << std::endl;

  //can engine see this now?
  NUM_VERTICES = graph.num_vertices();
  graphlab::timer ti;
  
  // create engine to count the number of triangles
  dc.cout() << "Counting Triangles..." << std::endl;
  graphlab::synchronous_engine<triangle_count> engine(dc, graph, clopts);
  engine.signal_all();
  engine.start();

  dc.cout() << "Counted in " << ti.current_time() << " seconds" << std::endl;

  if (PER_VERTEX_COUNT == false) {
    // size_t count = graph.map_reduce_edges<size_t>(get_edge_data);
    // dc.cout() << count << " Triangles"  << std::endl;
    //fix if only global count
    //dc.cout() << "Global is not currently supported"  << std::endl;
	vertex_data_type global_counts = graph.map_reduce_vertices<vertex_data_type>(get_vertex_data);
    size_t denom = (NUM_VERTICES)*(NUM_VERTICES-1)*(NUM_VERTICES-2)*(NUM_VERTICES-3)/24.; //normalize by |V| choose 4
	//size_t denom = 1;
	dc.cout() << "denominator: " << denom << std::endl;
	dc.cout() << "Global count: " << global_counts.num_triangles/3 << "  " << global_counts.num_wedges/3 << "  " << global_counts.num_disc/2 << "  " << global_counts.num_empty/3 << "  " << std::endl;
	dc.cout() << "Global count (normalized): " << global_counts.num_triangles/(denom*3.) << "  " << global_counts.num_wedges/(denom*3.) << "  " << global_counts.num_disc/(denom*2.) << "  " << global_counts.num_empty/(denom*3.) << "  " << std::endl;
    //print counts and normalized?
    // // boost::vector<size_t,4> global_counts = graph.map_reduce_vertices< boost::vector<size_t,4> >(get_vertex_data);
    // vertex_data_type global_counts = graph.map_reduce_vertices<vertex_data_type>(get_vertex_data);
    // dc.cout() << "Global count: " << global_counts.data().num_triangles << "  " << global_counts.data().num_wedges << 
    //   "  " << global_counts.data().num_disc << "  " << global_counts.data().num_empty << "  " << std::endl;
  }
  else {
    graphlab::synchronous_engine<get_per_vertex_count> engine(dc, graph, clopts);
    engine.signal_all();
    engine.start();
    graph.save(per_vertex,
            save_triangle_count(),
            false, /* no compression */
            true, /* save vertex */
            false, /* do not save edge */
            1); /* one file per machine */

  }
  
  graphlab::stop_metric_server();

  graphlab::mpi_tools::finalize();
  return EXIT_SUCCESS;
} // End of main

