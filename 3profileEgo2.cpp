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
#include <limits>

//comment the following line if you want to use integer counters
#define  DOUBLE_COUNTERS

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
double sample_prob_keep = 1; // This is the global sampling probability of an edge
double min_prob = 1;
double max_prob = 1;
double prob_step = 0.5;
size_t total_edges = 0;
int sample_iter = 1;
double sample_prob2=0.9;// This is the sample probabiltiy for edges in the Ego neighborhood
size_t ne_thresh=200;

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

size_t HASH_THRESHOLD = 1000000000;//64;

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
static size_t count_set_intersect(
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


//  CLIQUE FINDING CHANGES - NEW STRUCTURE.
struct twoids{
   graphlab::vertex_id_type first;
   graphlab::vertex_id_type second;
 void save(graphlab::oarchive &oarc) const {
    oarc << first << second; }
  void load(graphlab::iarchive &iarc) {
    iarc >> first>>second; }

};


/*
 * Each vertex maintains a list of all its neighbors.
 * and a final count for the number of triangles it is involved in
 */
struct vertex_data_type {
  //vertex_data_type(): num_triangles(0){ }
  // A list of all its neighbors
  vid_vector vid_set;
  vid_vector vid_set_sampled;//with sample_prob2
  // The number of triangles this vertex is involved it.
  // only used if "per vertex counting" is used
#ifdef DOUBLE_COUNTERS
  double num_triangles;
  double num_wedges;
  double num_wedges_e;
  double num_wedges_c;
  //solution to local equations  
  double num_disc;
  double num_empty;
#else
  size_t num_triangles;
  size_t num_wedges;
  size_t num_wedges_e;
  size_t num_wedges_c;
  //solution to local equations
  size_t num_disc;
  size_t num_empty;
#endif
  std::vector<twoids> conn_neighbors;

  vertex_data_type& operator+=(const vertex_data_type& other) {
    num_triangles += other.num_triangles;
    num_wedges += other.num_wedges;
    num_wedges_e += other.num_wedges_e;
    num_wedges_c += other.num_wedges_c;
    num_disc += other.num_disc;
    num_empty += other.num_empty;
    return *this;
  }
  
  void save(graphlab::oarchive &oarc) const {
    oarc << conn_neighbors<< vid_set_sampled << vid_set << num_triangles << num_wedges << num_wedges_e << num_wedges_c << num_disc << num_empty;
  }
  void load(graphlab::iarchive &iarc) {
    iarc >> conn_neighbors >>vid_set_sampled>> vid_set >> num_triangles >> num_wedges >> num_wedges_e >> num_wedges_c >> num_disc >>num_empty;
  }
};



/*
 * Each edge is simply a counter of triangles
 *
 */
//typedef uint32_t edge_data_type;

//NEW EDGE DATA AND GATHER
struct edge_data_type {
  
#ifdef DOUBLE_COUNTERS
  double n3;
  double n2;
  double n2e;
  double n2c;
  double n1;
  double eqn10_const_src;
  double eqn10_const_tar;
#else
  size_t n3;
  size_t n2;
  size_t n2e;
  size_t n2c;
  size_t n1;
  size_t eqn10_const_src; 
  size_t eqn10_const_tar;
#endif
  bool sample_indicator;
  bool sample_indicator2;//this one is for Ego neighborhood
  void save(graphlab::oarchive &oarc) const {
    oarc << n1 << n2 << n2e << n2c << n3 << sample_indicator<<sample_indicator2<<eqn10_const_src<<eqn10_const_tar;
  }
  void load(graphlab::iarchive &iarc) {
    iarc >> n1 >> n2 >> n2e >> n2c >> n3 >> sample_indicator>>sample_indicator2>>eqn10_const_src>>eqn10_const_tar;
  }
};


//combine the second gather with the h10 gather?
struct edge_sum_h10_gather {

#ifdef DOUBLE_COUNTERS
  double n3;
  // double n2;
  double n2e;
  double n2c;
  // double n1;
  double n3_double;
  double n2c_double;
  double n2c_n3;
#else
  size_t n3;
  // size_t n2;
  size_t n2e;
  size_t n2c;
  // size_t n1;
  size_t n3_double;
  size_t n2c_double;
  size_t n2c_n3;
#endif
  std::vector<twoids> common;
  
  edge_sum_h10_gather& operator+=(const edge_sum_h10_gather& other) {
    n3 += other.n3;
    // n2 += other.n2;
    n2e += other.n2e;
    n2c += other.n2c;
    // n1 += other.n1;
    n3_double += other.n3_double;
    n2c_double += other.n2c_double;
    n2c_n3 += other.n2c_n3;
    
    std::vector<twoids> p=other.common;
    for (size_t i=0;i<p.size();i++){
       twoids ele;
       ele=p.at(i); 
      common.push_back(ele);
   }

    return *this;
  }

  // serialize
  void save(graphlab::oarchive& oarc) const {

    size_t num1=common.size();
    oarc<<num1;    
    for (size_t i=0;i<common.size();i++){
      oarc<<common.at(i);
    }
   
    oarc << n2c_n3 << n2c_double << n3_double << n2c << n2e << n3;
  }

  // deserialize
  void load(graphlab::iarchive& iarc) {
    
    size_t num1=0;
    common.clear();
    iarc>>num1;
     for(size_t a = 0; a < num1; a++){
        twoids ele;
        iarc >> ele;
        common.push_back(ele);
    }
    
    iarc >> n2c_n3 >> n2c_double >> n3_double >> n2c >> n2e >> n3;
  }
};

struct clique_gather {
#ifdef DOUBLE_COUNTERS
  double h10e;
#else
  size_t h10e;
#endif
  clique_gather& operator+=(const clique_gather& other) {
    h10e += other.h10e;
    return *this;
  }
  // serialize
  void save(graphlab::oarchive& oarc) const {
    oarc << h10e;
  }
  // deserialize
  void load(graphlab::iarchive& iarc) {
    iarc >> h10e;
  }
};

struct edge_sum_gather {

#ifdef DOUBLE_COUNTERS
  double n3;
  // double n2;
  double n2e;
  double n2c;
  // double n1;
  double n3_double;
  double n2c_double;
  double n2c_n3;
#else
  size_t n3;
  // size_t n2;
  size_t n2e;
  size_t n2c;
  // size_t n1;
  size_t n3_double;
  size_t n2c_double;
  size_t n2c_n3;
#endif
  edge_sum_gather& operator+=(const edge_sum_gather& other) {
    n3 += other.n3;
    // n2 += other.n2;
    n2e += other.n2e;
    n2c += other.n2c;
    // n1 += other.n1;
    n3_double += other.n3_double;
    n2c_double += other.n2c_double;
    n2c_n3 += other.n2c_n3;
    
    return *this;
  }

  // serialize
  void save(graphlab::oarchive& oarc) const {
    oarc << n2c_n3 << n2c_double << n3_double << n2c << n2e << n3;
  }

  // deserialize
  void load(graphlab::iarchive& iarc) {
    iarc >> n2c_n3 >> n2c_double >> n3_double >> n2c >> n2e >> n3;
  }
};

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
 graphlab::vertex_id_type v1;  // These two are used to signal edges after sampling for Ego neighborrhood.
 std::vector<graphlab::vertex_id_type> vec1;

  graphlab::vertex_id_type v;
  std::vector<graphlab::vertex_id_type> vid_vec;

 // set_union_gather():v(-1) {
 // }
//NOT SURE IF THIS CONSTRUCTOR IS NEEDED OR NOT ELSEWHERE

 // size_t size() const {
 //   if (v == (graphlab::vertex_id_type)-1) return vid_vec.size();
 //   else return 1;
 // }
  /*
   * Combining with another collection of vertices.
   * Union it into the current set.
   */
  set_union_gather& operator+=(const set_union_gather& other) {
     size_t flag1=0,flag2=0; 
   if ((vid_vec.size() == 0) &&(v==(graphlab::vertex_id_type)-1) ) {
      (*this).v = other.v;
      (*this).vid_vec=other.vid_vec;
      flag1=1;
    }
   else if ((other.vid_vec.size() == 0) &&(other.v==(graphlab::vertex_id_type)-1) )
      flag1=1;

     if ((vec1.size() == 0) &&(v1==(graphlab::vertex_id_type)-1) ) {
      (*this).v1 = other.v1;
      (*this).vec1=other.vec1;
      flag2=1;
    }
   else if ((other.vec1.size() == 0) &&(other.v1==(graphlab::vertex_id_type)-1) )
      flag2=1;
 
 
   if(flag1==0) { 
     
    if(vid_vec.size()==0) {
     vid_vec.push_back(v);
     v= (graphlab::vertex_id_type)(-1);
    }
 
     if (other.vid_vec.size() > 0) {
      size_t ct = vid_vec.size();
      vid_vec.resize(vid_vec.size() + other.vid_vec.size());
      for (size_t i = 0; i < other.vid_vec.size(); ++i) {
        vid_vec[ct + i] = other.vid_vec[i];
      }
    }
    else if (other.v != (graphlab::vertex_id_type)-1) {
      vid_vec.push_back(other.v);
    }
  }
//REPLICATED THE ABOVE FOR vec1 and v1 variables that were added above.

  if(flag2==0) {  
   
   if(vec1.size()==0) {
     vec1.push_back(v1);
     v1= (graphlab::vertex_id_type)(-1);
    }
 
    if (other.vec1.size() > 0) {
      size_t ct = vec1.size();
      vec1.resize(vec1.size() + other.vec1.size());
      for (size_t i = 0; i < other.vec1.size(); ++i) {
        vec1[ct + i] = other.vec1[i];
      }
    }
    else if (other.v1 != (graphlab::vertex_id_type)-1) {
      vec1.push_back(other.v1);
    }

  } 

    return *this;
  }
  
  // serialize
  void save(graphlab::oarchive& oarc) const {
    oarc << bool(vid_vec.size() == 0);
    if (vid_vec.size() == 0) oarc << v;
    else oarc << vid_vec;
   
    oarc << bool(vec1.size() == 0);
    if (vec1.size() == 0) oarc << v1;
    else oarc << vec1;
 
  }

  // deserialize
  void load(graphlab::iarchive& iarc) {
    bool novvec;
    v = (graphlab::vertex_id_type)(-1);
    vid_vec.clear();
    iarc >> novvec;
    if (novvec) iarc >> v;
    else iarc >> vid_vec;

    bool novvec1;
    v1 = (graphlab::vertex_id_type)(-1);
    vec1.clear();
    iarc >> novvec1;
    if (novvec1) iarc >> v1;
    else iarc >> vec1;

  }
};

/*
 * Define the type of the graph
 */
typedef graphlab::distributed_graph<vertex_data_type,
                                    edge_data_type> graph_type;


//move init outside constructor (must be declared after graph_type)
void init_vertex(graph_type::vertex_type& vertex) { 
  vertex.data().num_triangles = 0; 
  vertex.data().num_wedges = 0;
  vertex.data().num_wedges_e = 0;
  vertex.data().num_wedges_c = 0; 
  vertex.data().num_disc = 0; 
  vertex.data().num_empty = 0;
}


void sample_edge(graph_type::edge_type& edge) {
  
  if(graphlab::random::rand01() < sample_prob_keep)   
    edge.data().sample_indicator = 1;
  else
    edge.data().sample_indicator = 0;
// The ego neighborhood sampling
   if(graphlab::random::rand01() < sample_prob2)   
    edge.data().sample_indicator2 = 1;
  else
    edge.data().sample_indicator2 = 0;

  
}


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
      gather.v = -1;
      gather.vid_vec.clear(); 
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
      gather.v = otherid; //will this work? what is v??
      gather.vid_vec.clear();
    //} 
   }

    if(edge.data().sample_indicator2 == 0){
      gather.v1 = -1; 
      gather.vec1.clear();
    }
    else{
      graphlab::vertex_id_type otherid = edge.target().id() == vertex.id() ?
                                       edge.source().id() : edge.target().id();
      gather.v1=otherid;
      gather.vec1.clear();
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

   if (neighborhood.vec1.size() == 0) {
     // neighborhood set may be empty or has only 1 element
     vertex.data().vid_set_sampled.clear();
     if (neighborhood.v1 != (graphlab::vertex_id_type(-1))) {
       vertex.data().vid_set_sampled.vid_vec.push_back(neighborhood.v1);
     }
   }
   else {
     vertex.data().vid_set_sampled.assign(neighborhood.vec1);
   }
   

  } // end of apply

  /*
   * Scatter over all edges to compute the intersection.
   * I only need to touch each edge once, so if I scatter just on the
   * out edges, that is sufficient.
   */
  edge_dir_type scatter_edges(icontext_type& context,
                              const vertex_type& vertex) const {
    if (do_not_scatter) return graphlab::NO_EDGES;
    else return graphlab::OUT_EDGES;
  }


  /*
   * For each edge, count the intersection of the neighborhood of the
   * adjacent vertices. This is the number of triangles this edge is involved
   * in.
   */
  void scatter(icontext_type& context,
              const vertex_type& vertex,
              edge_type& edge) const {
    //    vertex_type othervtx = edge.target();
    if (edge.data().sample_indicator == 1){
      const vertex_data_type& srclist = edge.source().data();
      const vertex_data_type& targetlist = edge.target().data();
      size_t tmp = 0;
      if (targetlist.vid_set.size() < srclist.vid_set.size()) {
        //edge.data() += count_set_intersect(targetlist.vid_set, srclist.vid_set);
        //will this work with += increment??
        tmp = count_set_intersect(targetlist.vid_set, srclist.vid_set);
      }
      else {
        //edge.data() += count_set_intersect(srclist.vid_set, targetlist.vid_set);
        tmp = count_set_intersect(srclist.vid_set, targetlist.vid_set);
      }
      // tmp2 = srclist.vid_set.size() + targetlist.vid_set.size();
      edge.data().n3 = tmp;
      // edge.data().n2 =  tmp2 - 2*tmp;
      
      edge.data().n2c = srclist.vid_set.size() - tmp - 1;
      edge.data().n2e = targetlist.vid_set.size() - tmp - 1;       

      // edge.data().n1 = context.num_vertices() - (tmp2 - tmp);
    }
  }
};

/*
 * This class is used in a second engine call if per vertex counts are needed.
 * The number of triangles a vertex is involved in can be computed easily
 * by summing over the number of triangles each adjacent edge is involved in
 * and dividing by 2. 
 */
class get_per_vertex_count :
      // public graphlab::ivertex_program<graph_type, size_t>,
      // public graphlab::ivertex_program<graph_type, edge_sum_gather>,
      public graphlab::ivertex_program<graph_type, edge_sum_h10_gather>,
      /* I have no data. Just force it to POD */
      public graphlab::IS_POD_TYPE  {
public:
  // Gather on all edges
  edge_dir_type gather_edges(icontext_type& context,
                             const vertex_type& vertex) const {
    return graphlab::ALL_EDGES;
  }
  // We gather the number of triangles each edge is involved in
  // size_t gather(icontext_type& context,
  gather_type gather(icontext_type& context,
                     const vertex_type& vertex,
                     edge_type& edge) const {
    // edge_sum_gather gather;
    edge_sum_h10_gather gather; //combined gathers
    if (edge.data().sample_indicator == 1){
      //return edge.data();
      // gather.n1 = edge.data().n1;
      // gather.n2 = edge.data().n2;
      gather.n3 = edge.data().n3;
      if (vertex.id() == edge.source().id()){
        gather.n2e = edge.data().n2e;
        gather.n2c = edge.data().n2c;
      }
      else{
        gather.n2e = edge.data().n2c;
        gather.n2c = edge.data().n2e;
      }
      gather.n3_double = (edge.data().n3)*(edge.data().n3-1)/2;
      gather.n2c_double = (gather.n2c)*(gather.n2c-1)/2;
      gather.n2c_n3 = gather.n3*gather.n2c;
       
     std::vector<graphlab::vertex_id_type> srcne,trgne; 
     srcne.clear();
     trgne.clear();
     if(vertex.id()==edge.source().id()) {
      srcne = edge.source().data().vid_set.vid_vec;
      if(srcne.size()>ne_thresh)
      trgne = edge.target().data().vid_set_sampled.vid_vec;
      else
      trgne =edge.target().data().vid_set.vid_vec;
     } 
     else {
        if(trgne.size()>ne_thresh)
         srcne = edge.source().data().vid_set_sampled.vid_vec;
        else
         srcne =edge.source().data().vid_set.vid_vec;

       trgne = edge.target().data().vid_set.vid_vec;  
     }  
      std::vector<graphlab::vertex_id_type> interlist;
     // SLIGHT CHANGE USING SAMPLED LIST DEPENDING ON THE DIRECTION OF GATHER 
       interlist.clear();    
      sort(srcne.begin(),srcne.end());
      sort(trgne.begin(),trgne.end());
      gather.common.clear();      
      // interlist contains the intersection.  
      std::set_intersection(srcne.begin(),srcne.end(),trgne.begin(),trgne.end(),std::back_inserter(interlist));
     
      // std::cout<<"From the perspective of "<<vertex.id()<<std::endl;
      // std::cout<<"Common nes of "<<edge.source().id()<<" , "<< edge.target().id()<<std::endl;
      
      //if(interlist.size()==0) /* Michael change - let's clear the set in anycase before populating it */
      //gather.common.clear();          
     
      if(vertex.id() == edge.source().id() ){
        for(size_t i=0;i<interlist.size();i++){
        
       //  std::cout<<interlist.at(i)<<std::endl;

         if (edge.target().id()< interlist.at(i)){
            twoids ele;
            ele.first=edge.target().id();
            ele.second=interlist.at(i);
            gather.common.push_back(ele);
         }
       // For the accumulating src, if target < member of list then push (target,member). If inequality is other way, when src-> member edge  will take care- nodouble counts.
       }     

      }
      else {
        for(size_t i=0;i<interlist.size();i++){
         if (edge.source().id()< interlist.at(i)){
            twoids ele;
            ele.first=edge.source().id();
            ele.second=interlist.at(i);
            gather.common.push_back(ele);
         }
       // For the accumulating target, if src < member of list then push (src,member). If inequality is other way, when target-> member edge  will take care- nodouble counts.
       }     
     }

    }
    else{
      // gather.n1 = 0;
      // gather.n2 = 0;
      gather.n2e = 0;
      gather.n2c = 0;
      gather.n3 = 0;
      gather.common.clear();
    }
    return gather;
  }

  /* the gather result is the total sum of the number of triangles
   * each adjacent edge is involved in . Dividing by 2 gives the
   * desired result.
   */
  void apply(icontext_type& context, vertex_type& vertex,
             const gather_type& ecounts) {
    //vertex.data().num_triangles = num_triangles / 2;
    //vid_set.size() or vid_vec.size()
    vertex.data().num_triangles = ecounts.n3 / 2;
    //vertex.data().num_wedges = ecounts.n2 - ( pow(vertex.data().vid_set.size(),2) + 3*vertex.data().vid_set.size() )/2 +
      //  vertex.data().num_triangles;

    vertex.data().num_wedges_c = ecounts.n2c/2;
    // vertex.data().num_wedges_e = ecounts.n2e;
    // vertex.data().num_wedges = vertex.data().num_wedges_e + vertex.data().num_wedges_c;

    if(ecounts.common.size()==0)
      vertex.data().conn_neighbors.clear();
    else   
     vertex.data().conn_neighbors=ecounts.common;
    //do the rest of clique finding here?? 
    // if not, then add a scatter and move whats below into the apply of a new engine
    // size_t h10v = 0;


    //store gathers here temporarily
    vertex.data().num_wedges = ecounts.n3_double;
    vertex.data().num_disc = -2*ecounts.n3_double + ecounts.n2c_n3;
    vertex.data().num_empty = 2*ecounts.n2c_double + 2*ecounts.n3_double - ecounts.n2c_n3;
  }

  edge_dir_type scatter_edges(icontext_type& context,
                              const vertex_type& vertex) const {
    return graphlab::OUT_EDGES;
  }

  void scatter(icontext_type& context,
              const vertex_type& vertex,
              edge_type& edge) const {
    //    vertex_type othervtx = edge.target();
    if (edge.data().sample_indicator == 1){
      const vertex_data_type& srclist = edge.source().data();
      const vertex_data_type& targetlist = edge.target().data();
      std::vector<graphlab::vertex_id_type> srcne = edge.source().data().vid_set.vid_vec;
      std::vector<graphlab::vertex_id_type>  trgne = edge.target().data().vid_set.vid_vec;
       
      std::vector<graphlab::vertex_id_type> srcne_sampled=edge.source().data().vid_set_sampled.vid_vec;
      std::vector<graphlab::vertex_id_type> trgne_sampled=edge.target().data().vid_set_sampled.vid_vec; 
      std::vector<graphlab::vertex_id_type> interlist;
      std::vector<graphlab::vertex_id_type> interlist1; 
          interlist.clear();    
          interlist1.clear();
      sort(srcne.begin(),srcne.end());
      sort(trgne.begin(),trgne.end());
      sort(srcne_sampled.begin(),srcne_sampled.end());
      sort(trgne_sampled.begin(),trgne_sampled.end());
      edge.data().eqn10_const_src=0;
      edge.data().eqn10_const_tar=0;      
      // interlist contains the intersection.
      if(srcne.size()>ne_thresh)
      std::set_intersection(srcne.begin(),srcne.end(),trgne_sampled.begin(),trgne_sampled.end(),std::back_inserter(interlist));
      else
       std::set_intersection(srcne.begin(),srcne.end(),trgne.begin(),trgne.end(),std::back_inserter(interlist));
  
      if(trgne.size()>ne_thresh)
        std::set_intersection(srcne_sampled.begin(),srcne_sampled.end(),trgne.begin(),trgne.end(),std::back_inserter(interlist1));
      else
        std::set_intersection(srcne.begin(),srcne.end(),trgne.begin(),trgne.end(),std::back_inserter(interlist1));
  

      //Check for each pair of members if they have a common edge, if they do count.
      graphlab::hopscotch_set<graphlab::vertex_id_type> *our_cset; 
      our_cset = new graphlab::hopscotch_set<graphlab::vertex_id_type>(64);
      foreach (graphlab::vertex_id_type v, interlist) {
        our_cset->insert(v);
      }
      graphlab::hopscotch_set<graphlab::vertex_id_type> *our_cset1; 
      our_cset1 = new graphlab::hopscotch_set<graphlab::vertex_id_type>(64);
      foreach (graphlab::vertex_id_type v, interlist1) {
        our_cset1->insert(v);
      }

//std::cout << "*** Hopscotch init time (scatter): " << ti_temp1.current_time() << " sec" << std::endl;

     for(size_t k=0;k<srclist.conn_neighbors.size();k++) {
        //graphlab::vertex_id_type num1=srclist.conn_neighbors.at(k).first;
        //graphlab::vertex_id_type num2=srclist.conn_neighbors.at(k).second;
        size_t flag1=0;
        flag1 = our_cset->count(srclist.conn_neighbors.at(k).first) + our_cset->count(srclist.conn_neighbors.at(k).second);
        //flag1 =(size_t)(our_cset->find(num1)!=our_cset->end()) + (size_t)(our_cset->find(num2)!=our_cset->end());
        // for(size_t i=0;i<interlist.size();i++){
        //    if((interlist.at(i)==num1)||(interlist.at(i)==num2))
        //     flag1++;
        //}
        //if (2 == our_cset->count(srclist.conn_neighbors.at(k).first) + our_cset->count(srclist.conn_neighbors.at(k).second))
        if(flag1==2) 
        edge.data().eqn10_const_src++;
      

      //   if ( ((srclist.conn_neighbors.at(k).first==interlist.at(i))&&(srclist.conn_neighbors.at(k).second==interlist.at(j)))||((srclist.conn_neighbors.at(k).first==interlist.at(j)) && (srclist.conn_neighbors.at(k).second==interlist.at(i))) )
        //    edge.data().eqn10_const++;
        //  std::cout<<srclist.conn_neighbiors.at(k).first<<" with " <<srclist.conn_neighbors.at(k).second<<std::endl;      
     }
//std::cout << "*** Triple FOR time (scatter): " << ti_temp1.current_time() << " sec" << std::endl;
     for (size_t k=0;k<targetlist.conn_neighbors.size();k++) {
        size_t flag1=0;
        flag1 = our_cset1->count(targetlist.conn_neighbors.at(k).first) + our_cset1->count(targetlist.conn_neighbors.at(k).second);
        if(flag1==2) 
        edge.data().eqn10_const_tar++;

     }
   }
  }
};


class get_ego_final :
      // public graphlab::ivertex_program<graph_type, size_t>,
      public graphlab::ivertex_program<graph_type, clique_gather>,
      /* I have no data. Just force it to POD */
      public graphlab::IS_POD_TYPE  {
public:
  // Gather on all edges
  edge_dir_type gather_edges(icontext_type& context,
                             const vertex_type& vertex) const {
    return graphlab::ALL_EDGES;
  }

  //gather
  gather_type gather(icontext_type& context,
                     const vertex_type& vertex,
                     edge_type& edge) const {
  
  clique_gather gather;
     if (edge.data().sample_indicator == 1){
       if(vertex.id()==edge.source().id())
        gather.h10e = edge.data().eqn10_const_src;
      else
       gather.h10e=edge.data().eqn10_const_tar; 
      }
      else {
        gather.h10e = 0;
      }
      return gather;
  }
     
  //apply
  void apply(icontext_type& context, vertex_type& vertex,
             const gather_type& ecounts) {
  
    size_t h10 = ecounts.h10e/3; //eqch clique counted by all 3 incoming edges
    
    //matrix inverse here??
    /*0.333333333333333   0.333333333333333  -0.166666666666667  -1.000000000000000
                      0  -1.000000000000000   0.500000000000000   3.000000000000000
                      0   1.000000000000000                   0  -3.000000000000000
                      0                   0                   0   1.000000000000000*/

    //can get simpler system with just n3_double/n2c_double/n2c_n3 and n3*N(v), n2c*N(v)
    // size_t h6 = (2*ecounts.n2c_double + 2*ecounts.n3_double - ecounts.n2c_n3 - 6*h10v)/6.;
    // size_t h8 = (-2*ecounts.n3_double + ecounts.n2c_n3 + 6*h10v)/2.;
    // size_t h9 = ecounts.n3_double - 3*h10v;

    // vertex.data().num_triangles += h10;
    // vertex.data().num_wedges = vertex.data().num_wedges_c + (vertex.data().num_wedges - 3*h10);

    if(vertex.data().vid_set.size()>ne_thresh)
    vertex.data().num_triangles = h10/pow(sample_prob2,3);
    else
    vertex.data().num_triangles = h10; 
    //scaling to account for ego sampling
    vertex.data().num_wedges -= 3*h10;
    vertex.data().num_disc = (vertex.data().num_disc + 6*h10)/2;
    vertex.data().num_empty = (vertex.data().num_empty - 6*h10)/6;
    //print here instead of in main??
    vertex.data().vid_set.clear(); //still necessary??    
  }

  // No scatter
  edge_dir_type scatter_edges(icontext_type& context,
                             const vertex_type& vertex) const {
    return graphlab::NO_EDGES;
  }


};

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
struct save_profile_count{
  std::string save_vertex(graph_type::vertex_type v) { 
  //   double nt = v.data().num_triangles;
  //   double n_followed = v.num_out_edges();
  //   double n_following = v.num_in_edges();

    //print?
    #if 0
    std::cout << "Estimators for vertex " << v.id() << ": " << (v.data().num_triangles)/pow(sample_prob_keep, 3) << " "
    << (v.data().num_wedges)/pow(sample_prob_keep, 2) - (1-sample_prob_keep)*(v.data().num_triangles)/pow(sample_prob_keep, 3) << " "
    << (v.data().num_disc)/sample_prob_keep - (1-sample_prob_keep)*(v.data().num_wedges)/pow(sample_prob_keep, 2) << " "
    << (v.data().num_empty)-(1-sample_prob_keep)*(v.data().num_disc)/sample_prob_keep << std::endl;
    #endif
  //   return graphlab::tostr(v.id()) + "\t" +
  //          graphlab::tostr(nt) + "\t" +
  //          graphlab::tostr(n_followed) + "\t" + 
  //          graphlab::tostr(n_following) + "\n";
  //
  /* WE SHOULD SCALE THE LOCAL COUNTS WITH p_sample BEFORE WRITING TO FILE!!!*/
  return 
         graphlab::tostr(v.id()) + "\t" +
         // graphlab::tostr(v.data().num_triangles) + "\t" +
         // graphlab::tostr(v.data().num_wedges) + "\t" +
         // graphlab::tostr(v.data().num_disc) + "\t" +
         // graphlab::tostr(v.data().num_empty) + "\n";
         graphlab::tostr((v.data().num_triangles)/pow(sample_prob_keep, 3)) + "\t" +
         graphlab::tostr((v.data().num_wedges)/pow(sample_prob_keep, 2) - (1-sample_prob_keep)*(v.data().num_triangles)/pow(sample_prob_keep, 3)) + "\t" +
         graphlab::tostr((v.data().num_disc)/sample_prob_keep - (1-sample_prob_keep)*(v.data().num_wedges)/pow(sample_prob_keep, 2)) + "\t" +
         graphlab::tostr((v.data().num_empty)-(1-sample_prob_keep)*(v.data().num_disc)/sample_prob_keep) + '\n';
  
  }
  std::string save_edge(graph_type::edge_type e) {
    return "";
  }
};


int main(int argc, char** argv) {

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
 clopts.attach_option("ht", HASH_THRESHOLD,
                       "Above this size, hash sets are used");
  clopts.attach_option("per_vertex", per_vertex,
                       "If not empty, will count the number of "
                       "triangles each vertex belongs to and "
                       "save to file with prefix \"[per_vertex]\". "
                       "The algorithm used is slightly different "
                       "and thus will be a little slower");
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
clopts.attach_option("sample_prob2",sample_prob2,"Ego sampling prob");
clopts.attach_option("ne_thresh",ne_thresh,"Neighborhood ego sampling threshold");

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
    std::cout << "--multiple iterations for global count only\n";
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
    dc.cout() << "Counting Ego subgraphs..." << std::endl;
    graphlab::synchronous_engine<triangle_count> engine1(dc, graph, clopts);
    // engine_type engine(dc, graph, clopts);


    engine1.signal_all();
    engine1.start();


    //dc.cout() << "Round 1 Counted in " << ti.current_time() << " seconds" << std::endl;
    
    //Sanity check for total edges count and degrees
    //total_edges = graph.map_reduce_vertices<size_t>(get_vertex_degree)/2;  
    //dc.cout() << "Total edges counted (after sampling) using degrees:" << total_edges << std::endl;

    //cannot put second engine before conditional?
    //graphlab::timer ti2;
    
    graphlab::synchronous_engine<get_per_vertex_count> engine2(dc, graph, clopts);
    engine2.signal_all();
    engine2.start();


    graphlab::synchronous_engine<get_ego_final> engine3(dc, graph, clopts);
    engine3.signal_all();
    engine3.start();
    
    //dc.cout() << "Round 2 Counted in " << ti2.current_time() << " seconds" << std::endl;
    //dc.cout() << "Total Running time is: " << ti.current_time() << "seconds" << std::endl;
    
    //no global counts, just print/write each ego subgraph without rescaling
    //if (PER_VERTEX_COUNT == false) {
       vertex_data_type global_counts = graph.map_reduce_vertices<vertex_data_type>(get_vertex_data);

    //   //size_t denom = (graph.num_vertices()*(graph.num_vertices()-1)*(graph.num_vertices()-2))/6.; //normalize by |V| choose 3, THIS IS NOT ACCURATE!
    //   //size_t denom = 1;
    //   //dc.cout() << "denominator: " << denom << std::endl;
    // //  dc.cout() << "Global count: " << global_counts.num_triangles/3 << "  " << global_counts.num_wedges/3 << "  " << global_counts.num_disc/3 << "  " << global_counts.num_empty/3 << "  " << std::endl;
    //   //dc.cout() << "Global count (normalized): " << global_counts.num_triangles/(denom*3.) << "  " << global_counts.num_wedges/(denom*3.) << "  " << global_counts.num_disc/(denom*3.) << "  " << global_counts.num_empty/(denom*3.) << "  " << std::endl;
    //   dc.cout() << "Global count from estimators: " 
  	 //      << (global_counts.num_triangles/3)/pow(sample_prob_keep, 3) << " "
  	 //      << (global_counts.num_wedges/3)/pow(sample_prob_keep, 2) - (1-sample_prob_keep)*(global_counts.num_triangles/3)/pow(sample_prob_keep, 3) << " "
   	//       << (global_counts.num_disc/3)/sample_prob_keep - (1-sample_prob_keep)*(global_counts.num_wedges/3)/pow(sample_prob_keep, 2) << " "
  	 //      << (global_counts.num_empty/3)-(1-sample_prob_keep)*(global_counts.num_disc/3)/sample_prob_keep  << " "
  	 //      << std::endl;

      total_time = ti.current_time();
      dc.cout() << "Total runtime: " << total_time << "sec." << std::endl;
 // JUST printing global counts.     
      std::cout.precision(20);
      std::cout << "Global N_10:" <<global_counts.num_triangles/4 << std::endl; 
      std::cout << "Global N_9:" <<global_counts.num_wedges/2<< std::endl;
      std::cout << "Global N_8:" <<global_counts.num_disc<< std::endl;
      std::cout << "Global N_6:" <<global_counts.num_empty<< std::endl;

 

      std::ofstream myfile;
      char fname[30];
      sprintf(fname,"counts_3_egos2.txt");
      bool is_new_file = true;
      if (std::ifstream(fname)){
        is_new_file = false;
      }
      myfile.open (fname,std::fstream::in | std::fstream::out | std::fstream::app);
      if(is_new_file) myfile << "#graph\tsample_prob_keep\ttriangles\twedges\tdisc\tempty\truntime" << std::endl;
      myfile << prefix << "\t"
           << /*sample_prob_keep*/sample_prob2 << "\t"
	   << /*sample_prob_keep*/ne_thresh << "\t"
           << std::setprecision (6)
           << total_time
           << std::endl;

      myfile.close();

      sprintf(fname,"netw_3_egos2_%d.txt",dc.procid());
      myfile.open (fname,std::fstream::in | std::fstream::out | std::fstream::app);
  
      myfile << dc.network_bytes_sent() - reference_bytes <<"\n";

      myfile.close();

   // }
    //else {
   if (PER_VERTEX_COUNT == true) {
      graph.save(per_vertex,
              save_profile_count(),
              false, /* no compression */
              true, /* save vertex */
              false, /* do not save edge */
              1); /* one file per machine */
              // clopts.get_ncpus());

    }
    

  }//for iterations
  }//while min/max_prob

  //graphlab::stop_metric_server();

  graphlab::mpi_tools::finalize();

  return EXIT_SUCCESS;
} // End of main

