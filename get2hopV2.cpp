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
//#include <boost/multiprecision/cpp_int.hpp>
#include <graphlab.hpp>
#include <graphlab/ui/metrics_server.hpp>
#include <graphlab/util/hopscotch_set.hpp>
#include <graphlab/macros_def.hpp>

//using namespace boost::multiprecision;

/**
 *
 *  
 * In this program we implement the "hash-set" version of the
 * "edge-iterator" algorithm described in
 * 
 *    T. Schank. Algorithmic Aspects of Triangle-Based Network Analysis.
 *    Phd in computer science, University Karlsruhe, 2007.
 *
 * The procedure is quite straightforward:
 *   - each vertex maintains a list of all of its neighbors in a hash set.
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
 */
 

// Radix sort implementation from https://github.com/gorset/radix
// Thanks to Erik Gorset
//
/*
Copyright 2011 Erik Gorset. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are
permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of
conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list
of conditions and the following disclaimer in the documentation and/or other materials
provided with the distribution.

THIS SOFTWARE IS PROVIDED BY Erik Gorset ``AS IS'' AND ANY EXPRESS OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Erik Gorset OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are those of the
authors and should not be interpreted as representing official policies, either expressed
or implied, of Erik Gorset.
*/

//probability of keeping an edge in the edges sampling process
double sample_prob_keep = 1;
double min_prob = 1;
double max_prob = 1;
double prob_step = 0.5;
size_t total_edges = 0;
int sample_iter = 1;

void radix_sort(graphlab::vertex_id_type *array, int offset, int end, int shift) {
    int x, y;
    graphlab::vertex_id_type value, temp;
    int last[256] = { 0 }, pointer[256];

    for (x=offset; x<end; ++x) {
        ++last[(array[x] >> shift) & 0xFF];
    }

    last[0] += offset;
    pointer[0] = offset;
    for (x=1; x<256; ++x) {
        pointer[x] = last[x-1];
        last[x] += last[x-1];
    }

    for (x=0; x<256; ++x) {
        while (pointer[x] != last[x]) {
            value = array[pointer[x]];
            y = (value >> shift) & 0xFF;
            while (x != y) {
                temp = array[pointer[y]];
                array[pointer[y]++] = value;
                value = temp;
                y = (value >> shift) & 0xFF;
            }
            array[pointer[x]++] = value;
        }
    }

    if (shift > 0) {
        shift -= 8;
        for (x=0; x<256; ++x) {
            temp = x > 0 ? pointer[x] - pointer[x-1] : pointer[0] - offset;
            if (temp > 64) {
                radix_sort(array, pointer[x] - temp, pointer[x], shift);
            } else if (temp > 1) {
                std::sort(array + (pointer[x] - temp), array + pointer[x]);
                //insertion_sort(array, pointer[x] - temp, pointer[x]);
            }
        }
    }
}

size_t HASH_THRESHOLD = 64;

// We on each vertex, either a vector of sorted VIDs
// or a hash set (cuckoo hash) of VIDs.
// If the number of elements is greater than HASH_THRESHOLD,
// the hash set is used. Otherwise the vector is used.
struct vid_vector{
  std::vector<graphlab::vertex_id_type> vid_vec;
  graphlab::hopscotch_set<graphlab::vertex_id_type> *cset;
  vid_vector(): cset(NULL) { }
  vid_vector(const vid_vector& v):cset(NULL) {
    (*this) = v;
  }

  vid_vector& operator=(const vid_vector& v) {
    if (this == &v) return *this;
    vid_vec = v.vid_vec;
    if (v.cset != NULL) {
      // allocate the cuckoo set if the other side is using a cuckoo set
      // or clear if I alrady have one
      if (cset == NULL) {
        cset = new graphlab::hopscotch_set<graphlab::vertex_id_type>(HASH_THRESHOLD);
      }
      else {
        cset->clear();
      }
      (*cset) = *(v.cset);
    }
    else {
      // if the other side is not using a cuckoo set, lets not use a cuckoo set
      // either
      if (cset != NULL) {
        delete cset;
        cset = NULL;
      }
    }
    return *this;
  }

  ~vid_vector() {
    if (cset != NULL) delete cset;
  }

  // assigns a vector of vertex IDs to this storage.
  // this function will clear the contents of the vid_vector
  // and reconstruct it.
  // If the assigned values has length >= HASH_THRESHOLD,
  // we will allocate a cuckoo set to store it. Otherwise,
  // we just store a sorted vector
  void assign(const std::vector<graphlab::vertex_id_type>& vec) {
    clear();
    if (vec.size() >= HASH_THRESHOLD) {
        // move to cset
        cset = new graphlab::hopscotch_set<graphlab::vertex_id_type>(HASH_THRESHOLD);
        foreach (graphlab::vertex_id_type v, vec) {
          cset->insert(v);
        }
    }
    else {
      vid_vec = vec;
      if (vid_vec.size() > 64) {
        radix_sort(&(vid_vec[0]), 0, vid_vec.size(), 24);
      }
      else {
        std::sort(vid_vec.begin(), vid_vec.end());
      }
      std::vector<graphlab::vertex_id_type>::iterator new_end = std::unique(vid_vec.begin(),
                                               vid_vec.end());
      vid_vec.erase(new_end, vid_vec.end());
    }
  }

  //ARE DUPLICATES ALREADY REMOVED WHEN INSERTING TO cset?
  // //new function to keep old elements instead of replace them, maybe use in gather
  // //change to take a vid_vector structure instead of a vid_vec vector?
  // void push(const std::vector<graphlab::vertex_id_type>& vec) {
  //   if ((size() + vec.size()) >= HASH_THRESHOLD) {
  //       // move to cset
  //       //if own cset is null then move over vid_vec to cset
  //       if (cset == NULL) {
  //         cset = new graphlab::hopscotch_set<graphlab::vertex_id_type>(HASH_THRESHOLD);
  //         foreach (graphlab::vertex_id_type v, vid_vec) {
  //           cset->insert(v);
  //         }
  //       }
  //       //what about if vec is a vector or cset?
  //       foreach (graphlab::vertex_id_type v, vec) {
  //         cset->insert(v);
  //       }
  //     //worry about unique elements the same way???? will unique work on cset?
  //     //erase everything element before adding it?
  //   }
  //   else {
  //     vid_vec.push_back(vec);
  //     if (vid_vec.size() > 64) {
  //       radix_sort(&(vid_vec[0]), 0, vid_vec.size(), 24);
  //     }
  //     else {
  //       std::sort(vid_vec.begin(), vid_vec.end());
  //     }
  //     std::vector<graphlab::vertex_id_type>::iterator new_end = std::unique(vid_vec.begin(),
  //                                              vid_vec.end());
  //     vid_vec.erase(new_end, vid_vec.end());
  //   }
  // }

  //new iterator that works on both .ved_vec and ->cset ??
  
  void save(graphlab::oarchive& oarc) const {
    oarc << (cset != NULL);
    if (cset == NULL) oarc << vid_vec;
    else oarc << (*cset);
  }


  void clear() {
    vid_vec.clear();
    if (cset != NULL) {
      delete cset;
      cset = NULL;
    }
  }

  size_t size() const {
    return cset == NULL ? vid_vec.size() : cset->size();
  }

  void load(graphlab::iarchive& iarc) {
    clear();
    bool hascset;
    iarc >> hascset;
    if (!hascset) iarc >> vid_vec;
    else {
      cset = new graphlab::hopscotch_set<graphlab::vertex_id_type>(HASH_THRESHOLD);
      iarc >> (*cset);
    }
  }
};

/*
  A simple counting iterator which can be used as an insert iterator.
  but only counts the number of elements inserted. Useful for
  use with counting the size of an intersection using std::set_intersection
*/
template <typename T>
struct counting_inserter {
  size_t* i;
  counting_inserter(size_t* i):i(i) { }
  counting_inserter& operator++() {
    ++(*i);
    return *this;
  }
  void operator++(int) {
    ++(*i);
  }

  struct empty_val {
    empty_val operator=(const T&) { return empty_val(); }
  };

  empty_val operator*() {
    return empty_val();
  }

  typedef empty_val reference;
};


/*
 * Computes the size of the intersection of two vid_vector's
 */
// static uint32_t count_set_intersect(
/*static size_t count_set_intersect(
             const vid_vector& smaller_set,
             const vid_vector& larger_set) {

  if (smaller_set.cset == NULL && larger_set.cset == NULL) {
    size_t i = 0;
    counting_inserter<graphlab::vertex_id_type> iter(&i);
    std::set_intersection(smaller_set.vid_vec.begin(), smaller_set.vid_vec.end(),
                          larger_set.vid_vec.begin(), larger_set.vid_vec.end(),
                          iter);
    return i;
  }
  else if (smaller_set.cset == NULL && larger_set.cset != NULL) {
    size_t i = 0;
    foreach(graphlab::vertex_id_type vid, smaller_set.vid_vec) {
      i += larger_set.cset->count(vid);
    }
    return i;
  }
  else if (smaller_set.cset != NULL && larger_set.cset == NULL) {
    size_t i = 0;
    foreach(graphlab::vertex_id_type vid, larger_set.vid_vec) {
      i += smaller_set.cset->count(vid);
    }
    return i;
  }
  else {
    size_t i = 0;
    foreach(graphlab::vertex_id_type vid, *(smaller_set.cset)) {
      i += larger_set.cset->count(vid);
    }
    return i;

  }
}

//will this work??
static size_t get_set_intersect(
             const vid_vector& smaller_set,
             const vid_vector& larger_set) {

  if (smaller_set.cset == NULL && larger_set.cset == NULL) {
    size_t i = 0;
    counting_inserter<graphlab::vertex_id_type> iter(&i);
    std::set_intersection(smaller_set.vid_vec.begin(), smaller_set.vid_vec.end(),
                          larger_set.vid_vec.begin(), larger_set.vid_vec.end(),
                          iter);
    // return i;
    return iter;
  }
  else if (smaller_set.cset == NULL && larger_set.cset != NULL) {
    //if == 1 then insert to the list...
    size_t i = 0;
    foreach(graphlab::vertex_id_type vid, smaller_set.vid_vec) {
      i += larger_set.cset->count(vid);
    }
    return i;
  }
  else if (smaller_set.cset != NULL && larger_set.cset == NULL) {
    size_t i = 0;
    foreach(graphlab::vertex_id_type vid, larger_set.vid_vec) {
      i += smaller_set.cset->count(vid);
    }
    return i;
  }
  else {
    size_t i = 0;
    foreach(graphlab::vertex_id_type vid, *(smaller_set.cset)) {
      i += larger_set.cset->count(vid);
    }
    return i;

  }
}*/


/*
 * Each vertex maintains a list of all its neighbors.
 * and a final count for the number of triangles it is involved in
 */
struct vertex_data_type {
  //vertex_data_type(): num_triangles(0){ }
  // A list of all its neighbors
  vid_vector vid_set;
  vid_vector two_hop_set; //things that are at most 2 hops away, can always remove 1 hop afterwards?
  // The number of triangles this vertex is involved it.
  // only used if "per vertex counting" is used
  // double num_triangles;
  // double num_wedges;
  // double num_wedges_e;
  // double num_wedges_c;  
  // double num_disc;
  // double num_empty;
  
  //maybe union operator here? neighborhoods from different edge pivots?

  // vertex_data_type& operator+=(const vertex_data_type& other) {
  //   num_triangles += other.num_triangles;
  //   num_wedges += other.num_wedges;
  //   num_wedges_e += other.num_wedges_e;
  //   num_wedges_c += other.num_wedges_c;
  //   num_disc += other.num_disc;
  //   num_empty += other.num_empty;
  //   return *this;
  // }
  
  void save(graphlab::oarchive &oarc) const {
    oarc << vid_set << two_hop_set;
  }
  void load(graphlab::iarchive &iarc) {
    iarc >> vid_set >> two_hop_set;
  }
};



/*
 * Each edge is simply a counter of triangles
 *
 */
//typedef uint32_t edge_data_type;

//NEW EDGE DATA AND GATHER
struct edge_data_type {
  double n3;
  double n2;
  double n2e;
  double n2c;
  double n1;
  bool sample_indicator;
  void save(graphlab::oarchive &oarc) const {
    //oarc << vid_set << num_triangles;
    oarc << n1 << n2 << n2e << n2c << n3 << sample_indicator;
  }
  void load(graphlab::iarchive &iarc) {
    iarc >> n1 >> n2 >> n2e >> n2c >> n3 >> sample_indicator;
  }
};

// struct edge_sum_gather {
//   double n3;
//   double n2;
//   double n2e;
//   double n2c;
//   double n1;
//   edge_sum_gather& operator+=(const edge_sum_gather& other) {
//     n3 += other.n3;
//     n2 += other.n2;
//     n2e += other.n2e;
//     n2c += other.n2c;
//     n1 += other.n1;
    
//     return *this;
//   }

//   // serialize
//   void save(graphlab::oarchive& oarc) const {
//     oarc << n1 << n2 << n2e << n2c << n3;
//   }

//   // deserialize
//   void load(graphlab::iarchive& iarc) {
//     iarc >> n1 >> n2 >> n2e >> n2c >> n3;
//   }
// };

// To collect the set of neighbors, we need a message type which is
// basically a set of vertex IDs

bool PER_VERTEX_COUNT = false;


/*
 * This is the gathering type which accumulates an array of
 * all neighboring vertices.
 * It is a simple wrapper around a vector with
 * an operator+= which simply performs a  +=
 */
struct set_union_gather {
  graphlab::vertex_id_type v;
  std::vector<graphlab::vertex_id_type> vid_vec;
  
  set_union_gather():v(-1) {
  }

  size_t size() const {
    if (v == (graphlab::vertex_id_type)-1) return vid_vec.size();
    else return 1;
  }
  /*
   * Combining with another collection of vertices.
   * Union it into the current set.
   */
  set_union_gather& operator+=(const set_union_gather& other) {
    // std::cout << "CURRENT SIZE: " << vid_vec.size() << " INCOMING SIZE: " << other.vid_vec.size() << "\n"; //WHY IS THIS 0??
    if (size() == 0) {
      (*this) = other;
      return (*this);
    }
    else if (other.size() == 0) {
      return *this;
    }

    if (vid_vec.size() == 0) {
      vid_vec.push_back(v);
      v = (graphlab::vertex_id_type)(-1);
    }
    //does this check for duplicates??
    if (other.vid_vec.size() > 0) {
      size_t ct = vid_vec.size();
      vid_vec.resize(vid_vec.size() + other.vid_vec.size());
      // std::cout << "OLD SIZE: " << ct << " NEW SIZE: " << vid_vec.size() << "\n";
      for (size_t i = 0; i < other.vid_vec.size(); ++i) {
        vid_vec[ct + i] = other.vid_vec[i];
      }
    }
    else if (other.v != (graphlab::vertex_id_type)-1) {
      vid_vec.push_back(other.v);
    }
    return *this;
  }
  
  // serialize
  void save(graphlab::oarchive& oarc) const {
    oarc << bool(vid_vec.size() == 0);
    if (vid_vec.size() == 0) oarc << v;
    else oarc << vid_vec;
  }

  // deserialize
  void load(graphlab::iarchive& iarc) {
    bool novvec;
    v = (graphlab::vertex_id_type)(-1);
    vid_vec.clear();
    iarc >> novvec;
    if (novvec) iarc >> v;
    else iarc >> vid_vec;
  }
};

//another sets_union_gather to union 2 sets instead of vertex and set???
//no vertex_id_type
//or just overload this operator?
//vectors_union_gather??
// graphlab::hopscotch_set<graphlab::vertex_id_type> *cset;


/*
 * Define the type of the graph
 */
typedef graphlab::distributed_graph<vertex_data_type,
                                    edge_data_type> graph_type;


// //move init outside constructor (must be declared after graph_type)
void init_vertex(graph_type::vertex_type& vertex) { 
//   vertex.data().num_triangles = 0; 
//   vertex.data().num_wedges = 0;
//   vertex.data().num_wedges_e = 0;
//   vertex.data().num_wedges_c = 0; 
//   vertex.data().num_disc = 0; 
//   vertex.data().num_empty = 0;
     
     vertex.data().two_hop_set.clear();
     vertex.data().vid_set.clear();
}


  void sample_edge(graph_type::edge_type& edge) {
     if(graphlab::random::rand01() < sample_prob_keep)   
       edge.data().sample_indicator = 1;
     else
       edge.data().sample_indicator = 0;
  }


/*
 * This class implements the triangle counting algorithm as described in
 * the header. On gather, we accumulate a set of all adjacent vertices.
 * If per_vertex output is not necessary, we can use the optimization
 * where each vertex only accumulates neighbors with greater vertex IDs.
 */
class hop_count :
      public graphlab::ivertex_program<graph_type,
                                      set_union_gather>,
      /* I have no data. Just force it to POD */
      public graphlab::IS_POD_TYPE  {
public:
  bool do_not_scatter;

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
    if(edge.data().sample_indicator == 0){
       gather.v = -1; //default value
     }
     else{
      graphlab::vertex_id_type otherid = edge.target().id() == vertex.id() ?
                                       edge.source().id() : edge.target().id();

    // size_t other_nbrs = (edge.target().id() == vertex.id()) ?
    //     (edge.source().num_in_edges() + edge.source().num_out_edges()): 
    //     (edge.target().num_in_edges() + edge.target().num_out_edges());

    // size_t my_nbrs = vertex.num_in_edges() + vertex.num_out_edges();

    //if (PER_VERTEX_COUNT || (other_nbrs > my_nbrs) || (other_nbrs == my_nbrs && otherid > vertex.id())) {
    //if (PER_VERTEX_COUNT || otherid > vertex.id()) {
      // std::cout << "THIS ID: " << vertex.id() << " OTHER ID: " << otherid << "\n";
      gather.v = otherid; //will this work? what is v??
    //} 
   } 
   return gather;
  }

  /*
   * the gather result now contains the vertex IDs in the neighborhood.
   * store it on the vertex. 
   */
  void apply(icontext_type& context, vertex_type& vertex,
             const gather_type& neighborhood) {
   do_not_scatter = false;
   if (neighborhood.vid_vec.size() == 0) {
     // neighborhood set may be empty or has only 1 element
     vertex.data().vid_set.clear();
     if (neighborhood.v != (graphlab::vertex_id_type(-1))) {
       vertex.data().vid_set.vid_vec.push_back(neighborhood.v);
     }
   }
   else {
     vertex.data().vid_set.assign(neighborhood.vid_vec);
   }
   do_not_scatter = vertex.data().vid_set.size() == 0;
  } // end of apply

  /*
   * Scatter over all edges to compute the intersection.
   * I only need to touch each edge once, so if I scatter just on the
   * out edges, that is sufficient.
   */
   //change to no scatter
  edge_dir_type scatter_edges(icontext_type& context,
                              const vertex_type& vertex) const {
    // if (do_not_scatter) return graphlab::NO_EDGES;
    // else return graphlab::OUT_EDGES;
    return graphlab::NO_EDGES;
  }


  /*
   * For each edge, count the intersection of the neighborhood of the
   * adjacent vertices. This is the number of triangles this edge is involved
   * in.
   */
  // void scatter(icontext_type& context,
  //             const vertex_type& vertex,
  //             edge_type& edge) const {
  //   //    vertex_type othervtx = edge.target();
  //   // if (edge.data().sample_indicator == 1){
  //     const vertex_data_type& srclist = edge.source().data();
  //     const vertex_data_type& targetlist = edge.target().data();
  //     size_t tmp= 0, tmp2 = 0;
  //     if (targetlist.vid_set.size() < srclist.vid_set.size()) {
  //       //edge.data() += count_set_intersect(targetlist.vid_set, srclist.vid_set);
  //       //will this work with += increment??
  //       tmp = count_set_intersect(targetlist.vid_set, srclist.vid_set);
  //     }
  //     else {
  //       //edge.data() += count_set_intersect(srclist.vid_set, targetlist.vid_set);
  //       tmp = count_set_intersect(srclist.vid_set, targetlist.vid_set);
  //     }
  //     tmp2 = srclist.vid_set.size() + targetlist.vid_set.size();
  //     edge.data().n3 = tmp;
  //     edge.data().n2 =  tmp2 - 2*tmp;
      
  //     edge.data().n2c = srclist.vid_set.size() - tmp - 1;
  //     edge.data().n2e = targetlist.vid_set.size() - tmp - 1;       

  //     edge.data().n1 = context.num_vertices() - (tmp2 - tmp);
  //   // }
  // }
};


/* another class for 2 hop neighborhood, for each edge, exchange 1hop infos to get 2 hop info*/
//union_find util instead??
//right now this lists everything at most 2 hops away??
class two_hop_count :
public graphlab::ivertex_program<graph_type,
                                      set_union_gather>,
      /* I have no data. Just force it to POD */
      public graphlab::IS_POD_TYPE  {
public:
  bool do_not_scatter;

  // Gather on all edges
  edge_dir_type gather_edges(icontext_type& context,
                             const vertex_type& vertex) const {
    //sample edges here eventually, maybe by random edge.id() so consistent between the 2 engines?
    return graphlab::ALL_EDGES;
  } 

  /*
   * For each edge, figure out the ID of the "other" vertex
   * and accumulate a union of 1 hop neighborhoods.
   */
  gather_type gather(icontext_type& context,
                     const vertex_type& vertex,
                     edge_type& edge) const {
    set_union_gather gather;
    if(edge.data().sample_indicator == 0){
       gather.v = -1; 
    }
    else{
      vid_vector othernbh = edge.target().id() == vertex.id() ?
                                       edge.source().data().vid_set : edge.target().data().vid_set;

    // size_t other_nbrs = (edge.target().id() == vertex.id()) ?
    //     (edge.source().num_in_edges() + edge.source().num_out_edges()): 
    //     (edge.target().num_in_edges() + edge.target().num_out_edges());

    /// size_t my_nbrs = vertex.num_in_edges() + vertex.num_out_edges();

    //if (PER_VERTEX_COUNT || (other_nbrs > my_nbrs) || (other_nbrs == my_nbrs && otherid > vertex.id())) {
    //if (PER_VERTEX_COUNT || otherid > vertex.id()) {
      // std::cout << "2 HOP SIZE: " << othernbh.size() << "\n";
      // std::cout << "2 HOP VEC SIZE: " << othernbh.vid_vec.size() << "\n";
      if (othernbh.cset == NULL) {
        gather.vid_vec = othernbh.vid_vec; //will this work? what is v?? vid_vec regardless of cuckoo hash?
      }
      else {
        // std::cout << "gathering from hopscotch: " << "\n";
        //convert to a vector for gather and back to cset at the end? 
        //will this create a huge bottleneck?
        std::vector<graphlab::vertex_id_type> cs2v(othernbh.size(),0);
        size_t vs = 0;
        for (graphlab::hopscotch_set<graphlab::vertex_id_type>::iterator cit=othernbh.cset->begin(); 
              cit != othernbh.cset->end(); cit++) {
          cs2v[vs] = *cit;
          vs++;
        }
        gather.vid_vec = cs2v;
      }

    //} 
    } 
   return gather;
  }


  /*
   * the gather result now contains the vertex IDs in the neighborhood.
   * store it on the vertex. 
   */
  void apply(icontext_type& context, vertex_type& vertex,
             const gather_type& neighborhood) {
   do_not_scatter = false;
   // if (neighborhood.vid_vec.size() == 0) { //should not happen here...
   if (neighborhood.size() == 0) {
     // neighborhood set can handle more than 1 element?
     vertex.data().two_hop_set.clear();
     if (neighborhood.v != (graphlab::vertex_id_type(-1))) {
       vertex.data().two_hop_set.vid_vec.push_back(neighborhood.v);
     }
   }
   else {
     vertex.data().two_hop_set.assign(neighborhood.vid_vec);
   }
   //remove duplicates here, or does 
   // std::vector<graphlab::vertex_id_type>::iterator new_end = std::unique(vertex.data().vid_set.begin(),
   //                                         vertex.data().vid_set.end());
   // // vid_vec.erase(new_end, vid_vec.end());
   
   //did vid_set.assign already remove duplicates??????
   //only need to prevent vertex id itself, erase-remove idiom
   //we could also remove vertices that are already 1 hop away
  
   if (vertex.data().two_hop_set.cset == NULL) {
    vertex.data().two_hop_set.vid_vec.erase( std::remove( vertex.data().two_hop_set.vid_vec.begin(), 
        vertex.data().two_hop_set.vid_vec.end(), vertex.id() ), vertex.data().two_hop_set.vid_vec.end() ); 
   }
   else {
    vertex.data().two_hop_set.cset->erase(vertex.id());
    // vertex.data().two_hop_set.cset->insert(vertex.id()); //this does not seem to create duplicates...
    // std::cout << "first count is " << vertex.data().two_hop_set.cset->count(vertex.id()) << "\n";
    // vertex.data().two_hop_set.cset->insert(vertex.id()); //this does not seem to create duplicates...
    // std::cout << "second count is " << vertex.data().two_hop_set.cset->count(vertex.id()) << "\n";
    // vertex.data().two_hop_set.cset->insert(vertex.id()); //this does not seem to create duplicates...
    // std::cout << "third count is " << vertex.data().two_hop_set.cset->count(vertex.id()) << "\n";
   }
   
   // vertex.data().vid_set.resize(std::distance(vertex.data().vid_set.begin(),newend));
   do_not_scatter = vertex.data().two_hop_set.size() == 0;
  } // end of apply


  // No scatter
  edge_dir_type scatter_edges(icontext_type& context,
                             const vertex_type& vertex) const {
    return graphlab::NO_EDGES;
  }
  /*
   * Scatter over all edges to compute the intersection.
   * I only need to touch each edge once, so if I scatter just on the
   * out edges, that is sufficient.
   */
  // edge_dir_type scatter_edges(icontext_type& context,
  //                             const vertex_type& vertex) const {
  //   if (do_not_scatter) return graphlab::NO_EDGES;
  //   else return graphlab::OUT_EDGES;
  // }


  /*
   * For each edge, count the intersection of the neighborhood of the
   * adjacent vertices. This is the number of triangles this edge is involved
   * in.
   */
  // void scatter(icontext_type& context,
  //             const vertex_type& vertex,
  //             edge_type& edge) const {
  //   //    vertex_type othervtx = edge.target();
  //   if (edge.data().sample_indicator == 1){
  //     const vertex_data_type& srclist = edge.source().data();
  //     const vertex_data_type& targetlist = edge.target().data();
  //     size_t tmp= 0, tmp2 = 0;
  //     if (targetlist.vid_set.size() < srclist.vid_set.size()) {
  //       //edge.data() += count_set_intersect(targetlist.vid_set, srclist.vid_set);
  //       //will this work with += increment??
  //       tmp = count_set_intersect(targetlist.vid_set, srclist.vid_set);
  //     }
  //     else {
  //       //edge.data() += count_set_intersect(srclist.vid_set, targetlist.vid_set);
  //       tmp = count_set_intersect(srclist.vid_set, targetlist.vid_set);
  //     }
  //     tmp2 = srclist.vid_set.size() + targetlist.vid_set.size();
  //     edge.data().n3 = tmp;
  //     edge.data().n2 =  tmp2 - 2*tmp;
      
  //     edge.data().n2c = srclist.vid_set.size() - tmp - 1;
  //     edge.data().n2e = targetlist.vid_set.size() - tmp - 1;       

  //     edge.data().n1 = context.num_vertices() - (tmp2 - tmp);
  //   }
  // }
};



/*
 * This class is used in a second engine call if per vertex counts are needed.
 * The number of triangles a vertex is involved in can be computed easily
 * by summing over the number of triangles each adjacent edge is involved in
 * and dividing by 2. 
 */
// class get_per_vertex_count :
//       // public graphlab::ivertex_program<graph_type, size_t>,
//       public graphlab::ivertex_program<graph_type, edge_sum_gather>,
//       /* I have no data. Just force it to POD */
//       public graphlab::IS_POD_TYPE  {
// public:
//   // Gather on all edges
//   edge_dir_type gather_edges(icontext_type& context,
//                              const vertex_type& vertex) const {
//     return graphlab::ALL_EDGES;
//   }
//   // We gather the number of triangles each edge is involved in
//   // size_t gather(icontext_type& context,
//   gather_type gather(icontext_type& context,
//                      const vertex_type& vertex,
//                      edge_type& edge) const {
//     edge_sum_gather gather;
//     if (edge.data().sample_indicator == 1){
//       //return edge.data();
//       gather.n1 = edge.data().n1;
//       gather.n2 = edge.data().n2;
//       gather.n3 = edge.data().n3;
//       if (vertex.id() == edge.source().id()){
//         gather.n2e = edge.data().n2e;
//         gather.n2c = edge.data().n2c;
//       }
//       else{
//         gather.n2e = edge.data().n2c;
//         gather.n2c = edge.data().n2e;
//       }
//     }
//     else{
//       gather.n1 = 0;
//       gather.n2 = 0;
//       gather.n2e = 0;
//       gather.n2c = 0;
//       gather.n3 = 0;
//     }
//     return gather;
//   }

//    the gather result is the total sum of the number of triangles
//    * each adjacent edge is involved in . Dividing by 2 gives the
//    * desired result.
   
//   void apply(icontext_type& context, vertex_type& vertex,
//              const gather_type& ecounts) {
//     //vertex.data().num_triangles = num_triangles / 2;
//     //vid_set.size() or vid_vec.size()
//     vertex.data().num_triangles = ecounts.n3 / 2;
//     //vertex.data().num_wedges = ecounts.n2 - ( pow(vertex.data().vid_set.size(),2) + 3*vertex.data().vid_set.size() )/2 +
//       //  vertex.data().num_triangles;

//     vertex.data().num_wedges_c = ecounts.n2c/2;
//     vertex.data().num_wedges_e = ecounts.n2e;
//     vertex.data().num_wedges = vertex.data().num_wedges_e + vertex.data().num_wedges_c;

//     vertex.data().num_disc = ecounts.n1 + /*context.num_edges()*/total_edges - 3*vertex.data().num_triangles + pow(vertex.data().vid_set.size(),2) - ecounts.n2; //works for small example?????
//     vertex.data().num_empty = (context.num_vertices()  - 1)*(context.num_vertices() - 2)/2 - 
//         (vertex.data().num_triangles + vertex.data().num_wedges + vertex.data().num_disc);
//     vertex.data().vid_set.clear(); //still necessary??    
//   }

//   // No scatter
//   edge_dir_type scatter_edges(icontext_type& context,
//                              const vertex_type& vertex) const {
//     return graphlab::NO_EDGES;
//   }


// };

//typedef graphlab::synchronous_engine<triangle_count> engine_type;

/* Used to sum over all the edges in the graph in a
 * map_reduce_edges call
 * to get the total number of triangles
 */
// size_t get_edge_data(const graph_type::edge_type& e) {
//   return e.data();
// }

vertex_data_type get_vertex_data(const graph_type::vertex_type& v) {
  return v.data();
}

//For sanity check of total edges via total degree
//size_t get_vertex_degree(const graph_type::vertex_type& v){
//	return v.data().vid_set.size();
//}

  size_t get_edge_sample_indicator(const graph_type::edge_type& e){
           return e.data().sample_indicator;
   }

/*
 * A saver which saves a file where each line is a vid / # triangles pair
 */
struct save_neighborhoods{
  std::string save_vertex(graph_type::vertex_type v) { 
  //   double nt = v.data().num_triangles;
  //   double n_followed = v.num_out_edges();
  //   double n_following = v.num_in_edges();

  //   return graphlab::tostr(v.id()) + "\t" +
  //          graphlab::tostr(nt) + "\t" +
  //          graphlab::tostr(n_followed) + "\t" + 
  //          graphlab::tostr(n_following) + "\n";
  //
  
  // if ((v.data().vid_set.vid_vec.size() > 3) && (v.data().two_hop_set.vid_vec.size() > 3)) {
  //   // if (v.data().vid_set.size() > 0) {
  // return graphlab::tostr(v.id()) + "\n" +
  //        // for(int i=0; i<v.data().vid_set.size(); i++){
  //         "\t" + graphlab::tostr(v.data().vid_set.vid_vec[3]) + "\n" +
  //        // }
  //        // "\n" +
  //        // for(int i=0; i<v.data().two_hop_set.size(); i++){
  //         "\t" + graphlab::tostr(v.data().two_hop_set.vid_vec[3]) + "\n";
  //        // }
  //        // graphlab::tostr(v.data().num_triangles) + "\t" +
  //        // graphlab::tostr(v.data().num_wedges) + "\t" +
  //        // graphlab::tostr(v.data().num_disc) + "\t" +
  //        // graphlab::tostr(v.data().num_empty) + "\n";
  //       }
  //       else {return graphlab::tostr(v.id()) + "\t" + graphlab::tostr(v.data().vid_set.size()) + "\n";}
  
  // char buf[1000];
  std::string buf = graphlab::tostr(v.id()) + "\n";
  //print different things based on cset, ideally we should make an iterator for the class vid_vector
  if (v.data().vid_set.size() > 0) {
    buf += "1h: ";
    if (v.data().vid_set.cset == NULL) {
      for (size_t bt = 0; bt < v.data().vid_set.vid_vec.size(); bt++){
        buf += "\t" + graphlab::tostr(v.data().vid_set.vid_vec[bt]);
      }
    }
    else {
      for (graphlab::hopscotch_set<graphlab::vertex_id_type>::iterator bt=v.data().vid_set.cset->begin(); 
              bt != v.data().vid_set.cset->end(); bt++) {
        buf += "\t" + graphlab::tostr(*bt);
      }
    }
  }
  buf += "\n";

  if (v.data().two_hop_set.size() > 0) {
    buf += "2h: ";
    if (v.data().two_hop_set.cset == NULL) {
      for (size_t bt = 0; bt < v.data().two_hop_set.vid_vec.size(); bt++){
        buf += "\t" + graphlab::tostr(v.data().two_hop_set.vid_vec[bt]);
      }
    }
    else {
      for (graphlab::hopscotch_set<graphlab::vertex_id_type>::iterator bt=v.data().two_hop_set.cset->begin(); 
              bt != v.data().two_hop_set.cset->end(); bt++) {
        buf += "\t" + graphlab::tostr(*bt);
      }
    }
  }
  buf += "\n";
  
  return buf;
  }
  std::string save_edge(graph_type::edge_type e) {
    return "";
  }
};


int main(int argc, char** argv) {

  graphlab::command_line_options clopts("Exact 2 hop neighborhoods. "
    "The algorithm assumes that each undirected edge appears exactly once "
    "in the graph input. If edges may appear more than once, this procedure "
    "will over count.");
  std::string prefix, format;
  std::string per_vertex;
  clopts.attach_option("graph", prefix,
                       "Graph input. reads all graphs matching prefix*");
  clopts.attach_option("format", format,
                       "The graph format");
 clopts.attach_option("ht", HASH_THRESHOLD,
                       "Above this size, hash sets are used");
  clopts.attach_option("list_file", per_vertex,
                       "If not empty, will write the 1-hop and 2-hop"
                       "neighborhoods and "
                       "save to file with prefix \"[per_vertex]\". ");
 clopts.attach_option("sample_keep_prob", sample_prob_keep,
                        "Probability of keeping edge during sampling");
clopts.attach_option("sample_iter", sample_iter,
                       "Number of sampling iterations (global count)");
clopts.attach_option("min_prob", min_prob,
                       "min prob");
clopts.attach_option("max_prob", max_prob,
                       "max prob");
clopts.attach_option("prob_step", prob_step,
                       "prob step");

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

  if ((per_vertex != "") & (sample_iter > 1)) {
    std::cout << "--multiple iterations only when no output list\n";
    clopts.print_description();
    return EXIT_FAILURE;
  }

  if (per_vertex != "") PER_VERTEX_COUNT = true;
  // Initialize control plane using mpi
  graphlab::mpi_tools::init(argc, argv);
  graphlab::distributed_control dc;

  //graphlab::launch_metric_server();
  // load graph
  graph_type graph(dc, clopts);
  graph.load_format(prefix, format);
  graph.finalize();
  dc.cout() << "Number of vertices: " << graph.num_vertices() << std::endl
            << "Number of edges (before sampling):    " << graph.num_edges() << std::endl;

  dc.cout() << "sample_prob_keep = " << sample_prob_keep << std::endl;
  dc.cout() << "sample_iter = " << sample_iter << std::endl;

  size_t reference_bytes;
  double total_time;

  if(min_prob == max_prob){//i.e., they were not specified by user
    min_prob = sample_prob_keep;
    max_prob = sample_prob_keep;
  }

  double new_sample_prob = min_prob;
  while(new_sample_prob <= max_prob+0.00000000001){
    sample_prob_keep = new_sample_prob;
    new_sample_prob += prob_step;


  //START ITERATIONS HERE
  for (int sit = 0; sit < sample_iter; sit++) {

    reference_bytes = dc.network_bytes_sent();

    dc.cout() << "Iteration " << sit+1 << " of " << sample_iter << ". current sample prob: " << sample_prob_keep <<std::endl;

    graphlab::timer ti;
    
    // Initialize the vertex data
    graph.transform_vertices(init_vertex);

    //Sampling
    graph.transform_edges(sample_edge);
    //total_edges = graph.map_reduce_vertices<size_t>(get_vertex_degree)/2;
    total_edges = graph.map_reduce_edges<size_t>(get_edge_sample_indicator);
    dc.cout() << "Total edges counted (after sampling):" << total_edges << std::endl;


    // create engine to count the number of triangles
    dc.cout() << "Collecting hop 1..." << std::endl;
    graphlab::synchronous_engine<hop_count> engine1(dc, graph, clopts);
    // engine_type engine(dc, graph, clopts);


    engine1.signal_all();
    engine1.start();


    //dc.cout() << "Round 1 Counted in " << ti.current_time() << " seconds" << std::endl;
    
    //Sanity check for total edges count and degrees
    //total_edges = graph.map_reduce_vertices<size_t>(get_vertex_degree)/2;  
    //dc.cout() << "Total edges counted (after sampling) using degrees:" << total_edges << std::endl;

    //cannot put second engine before conditional?
    //graphlab::timer ti2;
    
    // graphlab::synchronous_engine<get_per_vertex_count> engine2(dc, graph, clopts);
    dc.cout() << "Collecting hop 2..." << std::endl;
    graphlab::synchronous_engine<two_hop_count> engine2(dc, graph, clopts);
    engine2.signal_all();
    engine2.start();
    //dc.cout() << "Round 2 Counted in " << ti2.current_time() << " seconds" << std::endl;
    //dc.cout() << "Total Running time is: " << ti.current_time() << "seconds" << std::endl;
    
    // if (PER_VERTEX_COUNT == false) {
      // vertex_data_type global_counts = graph.map_reduce_vertices<vertex_data_type>(get_vertex_data);

      // size_t denom = (graph.num_vertices()*(graph.num_vertices()-1)*(graph.num_vertices()-2))/6.; //normalize by |V| choose 3, THIS IS NOT ACCURATE!
      //size_t denom = 1;
      // dc.cout() << "denominator: " << denom << std::endl;
      //dc.cout() << "Global count: " << global_counts.num_triangles/3 << "  " << global_counts.num_wedges/3 << "  " << global_counts.num_disc/3 << "  " << global_counts.num_empty/3 << "  " << std::endl;
      //dc.cout() << "Global count (normalized): " << global_counts.num_triangles/(denom*3.) << "  " << global_counts.num_wedges/(denom*3.) << "  " << global_counts.num_disc/(denom*3.) << "  " << global_counts.num_empty/(denom*3.) << "  " << std::endl;
      // dc.cout() << "Global count from estimators: " 
  	   //    << (global_counts.num_triangles/3)/pow(sample_prob_keep, 3) << " "
  	   //    << (global_counts.num_wedges/3)/pow(sample_prob_keep, 2) - (global_counts.num_triangles/3)*(1-sample_prob_keep)/pow(sample_prob_keep, 3) << " "
   	  //     << (global_counts.num_disc/3)/sample_prob_keep - (global_counts.num_wedges/3)*(1-sample_prob_keep)/pow(sample_prob_keep, 2) << " "
  	   //    << (global_counts.num_empty/3)-(global_counts.num_disc/3)*(1-sample_prob_keep)/sample_prob_keep  << " "
  	   //    << std::endl;


      total_time = ti.current_time();
      dc.cout() << "Total runtime: " << total_time << "sec." << std::endl;

      std::ofstream myfile;
      char fname[25];
      sprintf(fname,"2_hop_list_times.txt");
      bool is_new_file = true;
      if (std::ifstream(fname)){
        is_new_file = false;
      }
      myfile.open (fname,std::fstream::in | std::fstream::out | std::fstream::app);
      if(is_new_file) myfile << "#graph\tsample_prob_keep\truntime" << std::endl;
      myfile << prefix << "\t"
             // << (global_counts.num_triangles/3)/pow(sample_prob_keep, 3) << "\t"
             // << (global_counts.num_wedges/3)/pow(sample_prob_keep, 2) - (global_counts.num_triangles/3)*(1-sample_prob_keep)/pow(sample_prob_keep, 3) << "\t"
             // << (global_counts.num_disc/3)/sample_prob_keep - (global_counts.num_wedges/3)*(1-sample_prob_keep)/pow(sample_prob_keep, 2) << "\t"
             // << (global_counts.num_empty/3)-(global_counts.num_disc/3)*(1-sample_prob_keep)/sample_prob_keep  << "\t"
             << sample_prob_keep << "\t"
             << std::setprecision (6)
             << total_time << "\t"
             << std::endl;

      myfile.close();

      sprintf(fname,"netw_2_hops_%d.txt",dc.procid());
      myfile.open (fname,std::fstream::in | std::fstream::out | std::fstream::app);

      myfile << dc.network_bytes_sent() - reference_bytes <<"\n";

      myfile.close();


    // }
    if (PER_VERTEX_COUNT==true) {
      graph.save(per_vertex,
              save_neighborhoods(),
              false, /* no compression */
              true, /* save vertex */
              false, /* do not save edge */
              1); /* one file per machine */
              // clopts.get_ncpus());

    }
    
    //dc.cout() << "Total Runtime: " << ti.current_time() << " sec" << std::endl;  

  }//for iterations
  }//while min/max_prob


  //graphlab::stop_metric_server();

  graphlab::mpi_tools::finalize();

  return EXIT_SUCCESS;
} // End of main

