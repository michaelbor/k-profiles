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
#include <boost/unordered_map.hpp> //either c++11 or boost library...
// #include <boost/intrusive/set.hpp>
// #include <boost/serialization/set.hpp>
//#include <boost/multiprecision/cpp_int.hpp>
#include <graphlab.hpp>
#include <graphlab/ui/metrics_server.hpp>
#include <graphlab/util/hopscotch_set.hpp>
#include <graphlab/macros_def.hpp>
#include <limits>
#include <list>
// list is to define lists of structures and sort the lists of structures. Useful for implementing Ethan's equation.

//comment the following line if you want to use integer counters
#define  DOUBLE_COUNTERS

//using namespace boost::multiprecision;
// using namespace boost::intrusive;

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
size_t total_vertices = 0;
#ifdef DOUBLE_COUNTERS
double n[4] = {};
#else
size_t n[4] = {};
#endif

// #define USIZE 11 
#define USIZE 17 

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

// size_t HASH_THRESHOLD = 64;
size_t HASH_THRESHOLD = 10000000000;

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


#ifdef DOUBLE_COUNTERS
class uvec {
public:
  double value[USIZE];
  double& operator[](int i){
    if( i > USIZE ){
      std::cout << "Index out of bounds" <<std::endl;
      return value[0];
    }
    return value[i];
  }
  const double& operator[](int i) const {
    if( i > USIZE ){
      std::cout << "Index out of bounds" <<std::endl;
      return value[0];
    }
    return value[i];
  }
  double operator*(const uvec& b) {
    double sum = 0;
    for (int i = 0; i < USIZE; ++i){
      sum += value[i]*b.value[i];
    }
    return sum;
  }
  //allow uvec * double?
  double operator*(const double* b) {
    double sum = 0;
    for (int i = 0; i < USIZE; ++i){
      sum += value[i]*(b[i]);
    }
    return sum;
  }
  uvec& operator+=(const uvec& b){
    for (int i = 0; i < USIZE; ++i){
      value[i] += b.value[i];
    }
    return *this;
  }
  
  void save(graphlab::oarchive& oarc) const {
    for (int i = 0; i < USIZE; ++i)
      oarc << value[i];
  }

  void load(graphlab::iarchive& iarc) {
    for(int i = 0; i < USIZE; ++i){
        iarc >> value[i];
    }
  }


};

#else
// The class hat is used to manipulate vectors written by Ethan.
class uvec {
private:
  // size_t value[USIZE];
public:
  size_t value[USIZE];
  size_t& operator[](int i){
    if( i > USIZE ){
      std::cout << "Index out of bounds" <<std::endl; 
      // return first element.
      return value[0];
    }
    return value[i];
  }
  const size_t& operator[](int i) const {
    if( i > USIZE ){
      std::cout << "Index out of bounds" <<std::endl; 
      // return first element.
      return value[0];
    }
    return value[i];
  }
  size_t operator*(const uvec& b) {
    size_t sum = 0;
    for (int i = 0; i < USIZE; ++i){
      sum += value[i]*b.value[i];
    }
    return sum;
  }
  
  // size_t operator*(const size_t& b) {
  //   size_t sum = 0;
  //   for (int i = 0; i < USIZE; ++i){
  //     sum += value[i]*b[i];
  //   }
  //   return sum;
  // }

  uvec& operator+=(const uvec& b){
    for (int i = 0; i < USIZE; ++i){
      value[i] += b.value[i];
    }
    return *this;
  }

  void save(graphlab::oarchive& oarc) const {
    for (int i = 0; i < USIZE; ++i)
      oarc << value[i];
  }

  void load(graphlab::iarchive& iarc) {
    for(int i = 0; i < USIZE; ++i){
        iarc >> value[i];
    }
  }


};
#endif


//  4-PROFILE CHANGES - NEW STRUCTURE.

struct idcount {
 graphlab::vertex_id_type vert_id;
 size_t count;
// Code to serialize and deserialize the structure. But not sure if this right ! 
 void save(graphlab::oarchive &oarc) const {
    oarc << vert_id << count; }
  void load(graphlab::iarchive &iarc) {
    iarc >> vert_id>>count; }
};
struct twoids{
   graphlab::vertex_id_type first;
   graphlab::vertex_id_type second;
 void save(graphlab::oarchive &oarc) const {
    oarc << first << second; }
  void load(graphlab::iarchive &iarc) {
    iarc >> first>>second; }

};

bool compare_idcount (const idcount& a, const idcount& b){	
 return (a.vert_id < b.vert_id);
}

// The compare_idcount funcion helps sort the structure. 

struct vertex_data_type {
  //vertex_data_type(): num_triangles(0){ }
  // A list of all its neighbors
  vid_vector vid_set;
  // The number of triangles this vertex is involved it.
  // only used if "per vertex counting" is used
#ifdef DOUBLE_COUNTERS
  double num_triangles;
  double num_wedges;
  double num_wedges_e;
  double num_wedges_c;  
  double num_disc;
  double num_empty;
  double num_disc_d; //4-PROFILE CHANGES -  n_1^d from the disconnected vertex perspective.
#else
  size_t num_triangles;
  size_t num_wedges;
  size_t num_wedges_e;
  size_t num_wedges_c;
  size_t num_disc;
  size_t num_empty;
  size_t num_disc_d;
#endif
  uvec u; 
  // double n4local[USIZE] = {}; // This is for calculating local 4-profiles. 
  std::vector<twoids> conn_neighbors;
  uvec n4local;

  vertex_data_type& operator+=(const vertex_data_type& other) {
    num_triangles += other.num_triangles;
    num_wedges += other.num_wedges;
    num_wedges_e += other.num_wedges_e;
    num_wedges_c += other.num_wedges_c;
    num_disc += other.num_disc;
    num_empty += other.num_empty;
    num_disc_d+=other.num_disc_d;
    for (int i=0; i<USIZE; i++){
      u[i] += other.u[i]; 
      n4local[i] += other.n4local[i];
    }
    return *this;
  }
  
  void save(graphlab::oarchive &oarc) const {
    oarc << conn_neighbors;
    oarc << vid_set << num_triangles << num_wedges << num_wedges_e << num_wedges_c << num_disc << num_empty<<num_disc_d << u << n4local;
  }
  void load(graphlab::iarchive &iarc) {
    iarc >> conn_neighbors;
    iarc >> vid_set >> num_triangles >> num_wedges >> num_wedges_e >> num_wedges_c >> num_disc >>num_empty>>num_disc_d >> u >> n4local;
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
  double eqn10_const;// new variable - 4- PROFILE for the 10th equation
#else
  size_t n3;
  size_t n2;
  size_t n2e;
  size_t n2c;
  size_t n1;
  long int eqn10_const; // not sure if the constant in H2,H7,H9,H10 equation can be
// negative or not
#endif

  bool sample_indicator;
  void save(graphlab::oarchive &oarc) const {
    oarc << n1 << n2 << n2e << n2c << n3 << sample_indicator << eqn10_const;
  }
  void load(graphlab::iarchive &iarc) {
    iarc >> n1 >> n2 >> n2e >> n2c >> n3 >> sample_indicator >> eqn10_const;
  }
};

// ADDED SOME VARIABLES THAT CAN HELP US GATHER EQS (34) till EQ(40)  - 7 MORE VARIABLES.


struct local4_gather {
#ifdef DOUBLE_COUNTERS
  double n3g;
  double n2eg;
  double n0g;
  double h10v;
#else
  size_t n3g;
  size_t n2eg;
  size_t n0g;
  size_t h10v;
#endif
  local4_gather& operator+=(const local4_gather& other) {
    n3g += other.n3g;
    n2eg += other.n2eg;
    n0g += other.n0g;
    h10v += other.h10v;
    //something for cliques here?
    return *this;
  }
  // serialize
  void save(graphlab::oarchive& oarc) const {
    oarc << n3g << n2eg << n0g << h10v;
  }

  void load(graphlab::iarchive& iarc) {
    iarc >> n3g >> n2eg >> n0g >> h10v;
  }
};


struct h10_gather{
// struct local4_gather {
  std::vector<twoids> common;
// To get common neighbor lists. Defining a vector of pair of vertex ids 
  h10_gather& operator+=(const h10_gather& other) {
    std::vector<twoids> p=other.common;
    // std::cout<<" Before merge "<<std::endl;
    // std::cout<<" Sizes are: "<<p.size()<<" "<<common.size()<<std::endl;
    for (size_t i=0;i<p.size();i++){
       twoids ele;
       ele=p.at(i); 
      common.push_back(ele);
    }

    // std::cout<<" After  merge "<<std::endl;
    // std::cout<<" Size is: "<<common.size()<<std::endl;

    return *this; 
 // simple merge. Just add entries of one to the other vector.  
  }
  // serialize- Updated serialize functions for vector called common
  void save(graphlab::oarchive& oarc) const {
    size_t num1=common.size();
    oarc<<num1;
    for (size_t i=0;i<common.size();i++){
      oarc<<common.at(i);
    }
    // oarc << n3g << n2eg << n0g;
  }

  void load(graphlab::iarchive& iarc) {
    size_t num1=0;
    common.clear();
    iarc>>num1;
     for(size_t a = 0; a < num1; a++){
        twoids ele;
        iarc >> ele;
        common.push_back(ele);
    }
    // iarc >> n3g >> n2eg >> n0g;
  } 
};


struct edge_sum_gather {
//CHANGE THESE BACK, OR USE FEWER TERMS AND COMBINE IN APPLY??
#ifdef DOUBLE_COUNTERS
  double n3;
  double n2;
  double n2e;
  double n2c;
  double n1;
  double n1_double;
  double n2c_double;
  double n2e_double;
  double n3_double;
  double n1_n2c;
  double n1_n2e;
  double n1_n3;
  double n2c_n2e;
  double n2c_n3;
  double n2e_n3;
  
#else
  size_t n3;
  size_t n2;
  size_t n2e;
  size_t n2c;
  size_t n1;
  size_t n1_double;
  size_t n2c_double;
  size_t n2e_double;
  size_t n3_double;
  size_t n1_n2c;
  size_t n1_n2e;
  size_t n1_n3;
  size_t n2c_n2e;
  size_t n2c_n3;
  size_t n2e_n3;
  
#endif
  // std::list<idcount> b; // 4-PROFILE CHANGE. This is a list of idcounts where each idcount stores the (id,count) for  accumulation during gather.
  std::vector<idcount> b; // which is better?
  // boost::unordered_map<graphlab::vertex_id_type,size_t> b; //ETHAN UNORDERED MAP
  // std::multiset<graphlab::vertex_id_type> b;
  
  edge_sum_gather& operator+=(const edge_sum_gather& other) {
    n3 += other.n3;
    n2 += other.n2;
    n2e += other.n2e;
    n2c += other.n2c;
    n1 += other.n1;
    n1_double+=other.n1_double;
    n2c_double+=other.n2c_double;
    n2e_double+=other.n2e_double;
    n3_double+=other.n3_double;
    n1_n2c+=other.n1_n2c;
    n1_n2e+=other.n1_n2e;
    n1_n3+=other.n1_n3;
    n2c_n2e+=other.n2c_n2e;
    n2c_n3+=other.n2c_n3;        
    n2e_n3+=other.n2e_n3; 
    
    //is this the correct merge??
    // 4-PROFILE CHANGES - PLEASE READ THIS AND VERIFY    
    // We need to add other.b and b.
   
    // First we sort both using the compare function.
   // std::list<idcount> e=other.b;
    std::vector<idcount> e=other.b;
   // This is because other is passed by a constant reference .So
    // it cannot be changed in this function.
  
   // b.sort(compare_idcount);    
   // e.sort(compare_idcount);
     
    std::sort(b.begin(),b.end(),compare_idcount);
    std::sort(e.begin(),e.end(),compare_idcount);

    // We initialize two iterators, that is used to traverse and  merge both (and add the counts possibly)
    
    //std::cout<<" Before += op "<< std::endl;
    //for (std::list<idcount>::iterator it3=b.begin();it3!=b.end();++it3){
     //  std::cout<<" vertex_id "<<it3->vert_id<<" count "<<it3->count<<std::endl;   }
  // std::cout<<std::endl;
 
 //for (std::list<idcount>::iterator it4=e.begin();it4!=e.end();++it4){
  //     std::cout<<" vertex_id "<<it4->vert_id<<" count "<<it4->count<<std::endl;   }
    

   //NEW MERGE CODE 
   /*   
    b.merge(e,compare_idcount); // e merged into b. But stuff with identical vertex ids are not merged. We will run a loop to merge.
    std::list<idcount>::iterator it; 
    std::list<idcount>::iterator it3;  
   
   
   for (it=b.begin(); it!=b.end();it++) {
     it3=it;//checking if the next element is same as this  
     it3++;
     if((it3!=b.end()) &&  (it3->vert_id== it->vert_id)) {
       it->count=it->count+it3->count;
       b.erase(it3); 
     }                   
   }
   */
    size_t num= b.size()+e.size();// This piece of code is required because generic merge has some different definition.
    idcount var;
    var.vert_id=0;
    var.count=0;  
    const idcount var1=var;
    std::vector<idcount> a(num,var1);   
    std::merge(b.begin(),b.end(),e.begin(),e.end(),a.begin(),compare_idcount);    
    
    std::vector<idcount>::iterator it; 
    std::vector<idcount>::iterator it3;  
   
   
    for (it=a.begin(); it!=a.end();it++) {
      it3=it;//checking if the next element is same as this  
      it3++;
      if((it3!=a.end()) &&  (it3->vert_id== it->vert_id)) {
        it->count=it->count+it3->count;
        a.erase(it3); 
      }                   
    }
    // Clear b and assign the merged a to b.
    b.clear();
    b=a;


    // //ETHAN UNORDERED MAP
    // //boost:: or std:: with c++11
    // // boost::unordered_map<vertex_id_type, size_t=0> e = other.b
    // boost::unordered_map<graphlab::vertex_id_type, size_t> e = other.b; //necessary to reassign here?
    // boost::unordered_map<graphlab::vertex_id_type, size_t>::iterator it4;
    //
    // //reserve b to size of e if e is larger than b???
    // for (it4 = e.begin(); it4 != e.end(); it4++) {
    // // while (it4 != e.end()) {
    //   //[] and insert can both invalidate iterators??!?
    //   // it4->first;
    //   // it4->second;
    //   //if b[second key] exists, add second value to its value
    //   //if not, insert second key with value second value (defaults to 0 then add)
    //   // b[it4->first] = b[it4->first] + it4->second;
    //   b[it4->first] += it4->second;
    //   //or use emplace/insert in the boost version??
    // }

    // //UNORDERED multiset or std::multiset
    // std::multiset<graphlab::vertex_id_type> e = other.b; //necessary to reassign here?
    // // boost::unordered_multiset<graphlab::vertex_id_type>::iterator it4;
    // b.insert(e.begin(), e.end());

    
    // OLD MERGE CODE.
   
  /*   std::list<idcount>::iterator it1,it2;
     it1=b.begin();
     it2=e.begin();
     if ((b.size()==0) && (e.size()==0)){
     b.clear();
     }
     else if (b.size()==0) {
     b=other.b; 
     }
     else if (e.size()==0){
      
     }
     else {
     int flag=0;
     while ((flag==0) )
     { 
       // std::cout<<"inside while \n";
  if (it1->vert_id==it2->vert_id){
          it1->count+=it2->count; // same vertex if merge and add the counts 
          if(it1!=b.end())
          it1++; // This must be incremented.           
         if (++it2==e.end()) {
         flag=1;
         }
         
       }
       else if (it1->vert_id < it2->vert_id){
        if (it1==b.end()) {
             idcount a=*it2;                  
             b.push_back(a); 
             it1=b.end();
             if (++it2==e.end()) {
                flag=1;
             }
              

           }
        else {
             it1++;     
             // b.insert(it1,*it2);  
           }   
       // if it1 is small, increment that   
       }
       else if (it1->vert_id > it2->vert_id){ 
            idcount a=*it2;
            b.insert(it1,a);
       // if it1 is bigger, then insert the smaller one.
           if (++it2==e.end()) {
             flag=1;
           }
           
  
       }
      

     }
   //std::cout<<" After += list op "<<std::endl; 
   //for (std::list<idcount>::iterator it3=b.begin();it3!=b.end();++it3){
   //     std::cout<<" vertex_id "<<it3->vert_id<<" count "<<it3->count<<std::endl; 
   //} 
   }
  */

    return *this;
  }

  // serialize
  void save(graphlab::oarchive& oarc) const {
    size_t num = b.size();
    oarc << num;
    // // std::list<idcount>::const_iterator it;
    std::vector<idcount>::const_iterator it;
    // std::multiset<graphlab::vertex_id_type>::const_iterator it;
    for (it = b.begin(); it != b.end(); ++it)
      oarc << *it;
    oarc << b;
    oarc << n1 << n2 << n2e << n2c << n3<<n1_double<<n2c_double<<n2e_double<<n3_double<<n1_n2c<<n1_n2e<<n1_n3<<n2c_n2e<<n2c_n3<<n2e_n3;
  }

  void load(graphlab::iarchive& iarc) {
    size_t num = 0;
    b.clear();
    iarc >> num;
    for(size_t a = 0; a < num; ++a){
	      idcount element;
        iarc >> element;
        b.push_back(element);
    }
    // size_t num = 0; //this is probably the bottleneck (and why vector is best), is there a faster way?
    // b.empty();
    // iarc >> num;
    // for(size_t a = 0; a < num; ++a){
    //     graphlab::vertex_id_type element;
    //     iarc >> element;
    //     b.insert(element);
    // }
    iarc >> b;
    iarc >> n1 >> n2 >> n2e >> n2c >> n3>>n1_double>>n2c_double>>n2e_double>>n3_double>>n1_n2c>>n1_n2e>>n1_n3>>n2c_n2e>>n2c_n3>>n2e_n3;
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
  vertex.data().num_disc_d=0;
  for (int i=0;i<USIZE;i++)
  {	vertex.data().u[i]=0;
  }
}


void sample_edge(graph_type::edge_type& edge) {
  
  edge.data().n3 = 0;
  edge.data().n2 = 0;
  edge.data().n2e = 0;
  edge.data().n2c = 0;
  edge.data().n1 = 0;
  
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
      size_t tmp= 0, tmp2 = 0;
      if (targetlist.vid_set.size() < srclist.vid_set.size()) {
        //edge.data() += count_set_intersect(targetlist.vid_set, srclist.vid_set);
        //will this work with += increment??
        tmp = count_set_intersect(targetlist.vid_set, srclist.vid_set);
      }
      else {
        //edge.data() += count_set_intersect(srclist.vid_set, targetlist.vid_set);
        tmp = count_set_intersect(srclist.vid_set, targetlist.vid_set);
      }
      tmp2 = srclist.vid_set.size() + targetlist.vid_set.size();
      edge.data().n3 = tmp;
      // edge.data().n2 =  tmp2 - 2*tmp;
      //reuse for a symmetric gather equation, why does this return inconsistent output?
      edge.data().n2 = (context.num_vertices() - srclist.vid_set.vid_vec.size() -1)*(srclist.vid_set.vid_vec.size() - 1)*
          (context.num_vertices() - targetlist.vid_set.vid_vec.size() -1)*(targetlist.vid_set.vid_vec.size() - 1);
      // std::cout << "symmetric edge way: " << context.num_vertices() - srclist.vid_set.vid_vec.size() -1 << ", " <<
      // (srclist.vid_set.vid_vec.size() - 1) << ", " << (context.num_vertices() - targetlist.vid_set.vid_vec.size() -1) << ", " <<
      // (targetlist.vid_set.vid_vec.size() - 1) << std::endl;
      //do this differently, like in 3profileV1

      edge.data().n2c = srclist.vid_set.size() - tmp - 1;
      edge.data().n2e = targetlist.vid_set.size() - tmp - 1;       

      edge.data().n1 = context.num_vertices() - (tmp2 - tmp);
    }
  }
};

/*
 * This class is used in a second engine call if per vertex counts are needed.
 * The number of triangles a vertex is involved in can be computed easily
 * by summing over the number of triangles each adjacent edge is involved in
 * and dividing by 2. 
 */
 //CHANGE THESE BACK AND CALC 13 VARS
class get_per_vertex_count :
      // public graphlab::ivertex_program<graph_type, size_t>,
      public graphlab::ivertex_program<graph_type, edge_sum_gather>,
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
    edge_sum_gather gather;

// 4 PROFILE CHANGES: ACTUALLY GATHERING FOR EQS (34) till (40) IN SCRATCH.


    if (edge.data().sample_indicator == 1){
      //return edge.data();
      gather.n1 = edge.data().n1;
      gather.n2 = edge.data().n2;
      gather.n3 = edge.data().n3;
      gather.n1_double = (edge.data().n1*(edge.data().n1-1))/2;
      
      // gather.n2c_double = (edge.data().n2c*(edge.data().n2c-1))/2;
      // gather.n2e_double = (edge.data().n2e*(edge.data().n2e-1))/2;
      
      gather.n3_double = (edge.data().n3*(edge.data().n3-1))/2;
      
      // gather.n1_n2c = edge.data().n1*edge.data().n2c;
      // gather.n1_n2e = edge.data().n1*edge.data().n2e;
      // gather.n2c_n3 = edge.data().n2c*edge.data().n3;
    
      gather.n1_n3 = edge.data().n1*edge.data().n3;
      gather.n2c_n2e = edge.data().n2c*edge.data().n2e;
      

      // std::cout << "edge.data().n2e = "<<edge.data().n2e<<" edge.data().n2c = "<<edge.data().n2c<<"\n";
      if (vertex.id() == edge.source().id()){
        gather.n2e = edge.data().n2e;
        gather.n2c = edge.data().n2c;
  	    gather.n2c_double = (edge.data().n2c*(edge.data().n2c-1))/2;
        gather.n2e_double = (edge.data().n2e*(edge.data().n2e-1))/2;
        gather.n1_n2c = edge.data().n1*edge.data().n2c;  
        gather.n1_n2e = edge.data().n1*edge.data().n2e;
        gather.n2c_n3 = edge.data().n2c*edge.data().n3;
        gather.n2e_n3 = edge.data().n2e*edge.data().n3;
        
        const vertex_data_type& targetlist = edge.target().data();
        for (size_t i=0; i<targetlist.vid_set.vid_vec.size();i++){
          // std::cout << " vertex "<<edge.target().id()<<" with neighbor "<<targetlist.vid_set.vid_vec.at(i)<<std::endl;
       //   std::cout <<"inside for 1 \n";
          if (targetlist.vid_set.vid_vec.at(i)!= edge.source().id()){
            idcount dum;
            dum.vert_id=targetlist.vid_set.vid_vec.at(i); // assigning the id of the structure to that of the ith neighbor.
            dum.count=1;
            gather.b.push_back(dum);// push it in the list
          } // doing this only for all neighbors of the target thats not the source. 
        }

        // //ETHAN UNORDERED MAP, pass to gather all at once??
        // for (size_t i=0; i<targetlist.vid_set.vid_vec.size();i++){
        //   if (targetlist.vid_set.vid_vec.at(i)!= edge.source().id()){
        //     // boost::unordered_map<graphlab::vertex_id_type,size_t> dum;
        //     // dum[targetlist.vid_set.vid_vec.at(i)] = 1;
        //     // gather.b.insert(dum);  //add key and value
        //     //will this work instead??
        //     gather.b[targetlist.vid_set.vid_vec.at(i)] = 1;
        //     // gather.b.insert(std::pair<graphlab::vertex_id_type,size_t>(targetlist.vid_set.vid_vec.at(i),1));
        //   }
        // }

        // //UNORDERED MULTISET
        // for (size_t i=0; i<targetlist.vid_set.vid_vec.size();i++){
        //   if (targetlist.vid_set.vid_vec.at(i)!= edge.source().id()){
        //     gather.b.insert(targetlist.vid_set.vid_vec.at(i));
        //   }
        // }

        // gather.n1_n2e_p = edge.data().n1*edge.data().n2e + 
        //   // (context.num_vertices() - edge.source().data().vid_set.size() - 1)*(edge.source().data().vid_set.size() - 1)*
        //   // (context.num_vertices() - targetlist.vid_set.size() - 1)*(targetlist.vid_set.size() - 1);
        //   edge.data().n2;
        //   // std::cout << "vertex way: " << (context.num_vertices() - edge.source().data().vid_set.size() - 1)*(edge.source().data().vid_set.size() - 1)*(context.num_vertices() - targetlist.vid_set.size() - 1)*(targetlist.vid_set.size() - 1)<<std::endl;
        //   // std::cout << "symmetric way: " << edge.data().n2 <<std::endl;
      }

      else{
        gather.n2e = edge.data().n2c;
        gather.n2c = edge.data().n2e;
        gather.n2c_double=(edge.data().n2e*(edge.data().n2e-1))/2;
        gather.n2e_double = (edge.data().n2c*(edge.data().n2c-1))/2;
        gather.n1_n2c=edge.data().n1*edge.data().n2e;
        gather.n1_n2e = edge.data().n1*edge.data().n2c;

        const vertex_data_type& srclist = edge.source().data();
        for (size_t i=0; i<srclist.vid_set.vid_vec.size();i++){
          //  std::cout <<"entering for 2 \n";
         //  std::cout << " vertex "<<edge.source().id()<<" with neighbor "<<srclist.vid_set.vid_vec.at(i)<<std::endl;
          if (srclist.vid_set.vid_vec.at(i)!= edge.target().id()){ 
            idcount dum;
            dum.vert_id=srclist.vid_set.vid_vec.at(i); // assigning the id of the structure to that of the ith neighbor.
            dum.count=1;
            gather.b.push_back(dum);// push it in the list in case num is not enough. 
           
         } // doing this only for all neighbors for the src that is not the target. 
       // std::cout <<"leaving for 2 \n";
        }

        // //ETHAN UNORDERED MAP, pass to gather all at once??
        // for (size_t i=0; i<srclist.vid_set.vid_vec.size();i++){
        //   if (srclist.vid_set.vid_vec.at(i)!= edge.target().id()){
        //    // gather.b.insert({{srclist.vid_set.vid_vec.at(i),1}});  //add key and value
        //     gather.b[srclist.vid_set.vid_vec.at(i)] = 1;
        //   // gather.b.insert(std::pair<graphlab::vertex_id_type,size_t>(srclist.vid_set.vid_vec.at(i),1));
        //   }
        // }

        // //UNORDERED MULTISET
        // for (size_t i=0; i<srclist.vid_set.vid_vec.size();i++){
        //   if (srclist.vid_set.vid_vec.at(i)!= edge.source().id()){
        //     gather.b.insert(srclist.vid_set.vid_vec.at(i));
        //   }
        // }

 	      gather.n2c_n3=edge.data().n2e*edge.data().n3;
        gather.n2e_n3=edge.data().n2c*edge.data().n3;
        //why is this different every time??
        // gather.n1_n2e_p = edge.data().n1*edge.data().n2c + 
        //   // (context.num_vertices() - srclist.vid_set.size() - 1)*(srclist.vid_set.size() - 1)*
        //   // (context.num_vertices() - edge.target().data().vid_set.size() - 1)*(edge.target().data().vid_set.size() - 1);
        //   edge.data().n2;
        //   // std::cout << "vertex way: " << edge.data().n1*edge.data().n2c + (context.num_vertices() - edge.target().data().vid_set.size() - 1)*(edge.target().data().vid_set.size() - 1)*(context.num_vertices() - srclist.vid_set.size() - 1)*(srclist.vid_set.size() - 1)<<std::endl;
        //   // std::cout << "vertex way (no hash): " << edge.data().n1*edge.data().n2c + (context.num_vertices() - edge.target().data().vid_set.vid_vec.size() - 1)*(edge.target().data().vid_set.vid_vec.size() - 1)*(context.num_vertices() - srclist.vid_set.vid_vec.size() - 1)*(srclist.vid_set.vid_vec.size() - 1)<<std::endl;
        //   // std::cout << "symmetric way: " << edge.data().n1*edge.data().n2c + edge.data().n2 <<std::endl;
      }
    }


    else{
      gather.n1 = 0;
      gather.n2 = 0;
      gather.n2e = 0;
      gather.n2c = 0;
      gather.n3 = 0;
      gather.n1_double=0;
      gather.n2c_double=0;
      gather.n2e_double=0;
      gather.n3_double=0;
      gather.n1_n2c=0;
      gather.n1_n2e=0;
      gather.n1_n3=0;
      gather.n2c_n2e=0;
      gather.n2c_n3=0;
      gather.n2e_n3=0;
    }
    return gather;
  }


//CHANGE THIS APPLY TO STORE 13 VARS

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
    vertex.data().num_wedges_e = ecounts.n2e;
    vertex.data().num_wedges = vertex.data().num_wedges_e + vertex.data().num_wedges_c;

    // vertex.data().num_disc = ecounts.n1 + context.num_edges()total_edges - 3*vertex.data().num_triangles + pow(vertex.data().vid_set.size(),2) - ecounts.n2; //works for small example?????
    // vertex.data().num_disc = ecounts.n1 + total_edges - vertex.data().vid_set.size() - vertex.data().num_triangles - vertex.data().num_wedges_e; //new eq??
    vertex.data().num_disc_d = total_edges - (vertex.data().vid_set.size() + vertex.data().num_wedges_e + vertex.data().num_triangles); 
    vertex.data().num_disc = ecounts.n1 + vertex.data().num_disc_d;
    vertex.data().num_empty = (context.num_vertices()  - 1)*(context.num_vertices() - 2)/2 - 
        (vertex.data().num_triangles + vertex.data().num_wedges + vertex.data().num_disc);
  // 4-PROFILE ADDITION  - EQUATIONS (34) till (40) excluding the OR equations - in the scracth pdf. 
     /*//this order once things are reduced globally, with 2hop as 8
     vertex.data().u[0]=ecounts.n1_double;
     vertex.data().u[1]=ecounts.n2c_double;
     vertex.data().u[2]=ecounts.n1_n2c;
     vertex.data().u[3]=ecounts.n1_n3;
     vertex.data().u[4]=ecounts.n2c_n2e; 
     vertex.data().u[5]=ecounts.n2c_n3;
     vertex.data().u[7]=ecounts.n3_double;

 // 4-PROFILE ADDITION - EQUATION (42) - calculating n_{1,v}^d and then using it for u[7]. 
     // // The above equation based on:  n_{1,v}^d= |E| - x. x=Gamma(v)+n_{2,v}^e+n_{3,v}-VERIFIED WITH MICHAEL. 
     vertex.data().u[6]=vertex.data().num_disc_d*vertex.data().vid_set.size(); 

     //now need n_{0,v}^d instead (43), simply non neighbors choose 2 times neighbors?
     //NO, but its just n_0!
     // vertex.data().u[8] = (context.num_vertices() - vertex.data().vid_set.size() - 1)* (context.num_vertices() - vertex.data().vid_set.size() - 2)/6*vertex.data().vid_set.size();
     // vertex.data().u[7] = vertex.data().num_empty * vertex.data().vid_set.size();
     */

     vertex.data().u[0]=ecounts.n1_double;
     vertex.data().u[1]=ecounts.n2c_double;
     vertex.data().u[2]=ecounts.n2e_double;
     vertex.data().u[3]=ecounts.n3_double;
     vertex.data().u[4]=ecounts.n1_n2c;
     vertex.data().u[5]=ecounts.n1_n2e;
     vertex.data().u[6]=ecounts.n1_n3;
     vertex.data().u[7]=ecounts.n2c_n2e; 
     vertex.data().u[8]=ecounts.n2c_n3;
     vertex.data().u[9]=ecounts.n2e_n3;
     vertex.data().u[10] = vertex.data().num_empty * vertex.data().vid_set.size();

 // Now using the gathered information from ecounts.b to form u[9].  So every id from there has to be checked if it is a member of the vertex's neighbor and only if not count choose 2 must be 
 //  aggregated.
     vertex.data().u[11]=0;
     // std::list<idcount> e1=ecounts.b;
     // std::list<idcount>::iterator it;
     
     std::vector<idcount> e1=ecounts.b;
     std::vector<idcount>::iterator it;

     for (it=e1.begin();it!=e1.end();++it){
        
       std::vector<graphlab::vertex_id_type> vec1 (1,it->vert_id); // initializing a vector with one id as element.
       vid_vector dummy; // create a vid_vector out of this vec 1 for uses with count_set_intersect
       dummy.assign(vec1);       
       if (  count_set_intersect(dummy,vertex.data().vid_set)== 0 ){
         vertex.data().u[11]+= (it->count * (it->count - 1))/2; // doing count choose 2 if the id is not the vertex neighbor.         
       }
 
     }

    // //ETHAN UNORDERED MAP
    // boost::unordered_map<graphlab::vertex_id_type,size_t> e1=ecounts.b;
    // boost::unordered_map<graphlab::vertex_id_type,size_t>::const_iterator it;   
    // for (it=e1.begin();it!=e1.end();++it){
    // // while (it!=e1.end()) {    
    //    std::vector<graphlab::vertex_id_type> vec1 (1,it->first); // initializing a vector with one id as element.
    //    vid_vector dummy; // create a vid_vector out of this vec 1 for uses with count_set_intersect
    //    dummy.assign(vec1);       
    //    if (  count_set_intersect(dummy,vertex.data().vid_set)== 0 ){
    //      vertex.data().u[11]+= (it->second * (it->second - 1))/2; // doing count choose 2 if the id is not the vertex neighbor.
    //     // std::cout<<" vertex "<< vertex.id() <<" non-neighbor vertex_id "<<it->vert_id<<" count "<<it->count<<std::endl;         
    //     }
    //  }

     //global n0g can be computed as num_empty*degree, since reduced globally anyway, uglobal[9]
     // vertex.data().u[12] = vertex.data().num_empty*vertex.data().vid_set.size();
     
  
    // //UNORDERED MULTISET??
    // std::multiset<graphlab::vertex_id_type> e1=ecounts.b;
    // std::multiset<graphlab::vertex_id_type>::const_iterator it;   
    // size_t pathcount;
    // //instead of iterate through unique elements, iterate over nonneighbors?
    // //are things 2 hops away less than nonneighbors? in most cases
    // for (it=e1.begin();it!=e1.end();++it){
    // // while (it!=e1.end()) {    
    //    std::vector<graphlab::vertex_id_type> vec1 (1,*it); // initializing a vector with one id as element.
    //    vid_vector dummy; // create a vid_vector out of this vec 1 for uses with count_set_intersect
    //    dummy.assign(vec1);       
    //    if (  count_set_intersect(dummy,vertex.data().vid_set)== 0 ){
    //      pathcount = e1.count(*it);
    //      vertex.data().u[9]+= (pathcount * (pathcount - 1))/2; // doing count choose 2 if the id is not the vertex neighbor.
    //      //increase the iterator to pass any duplicates
    //      std::advance(it,pathcount - 1);
    //     }
    //  }
     
  // KARTHIK CHANGE- COMMENTED THe FOLLOWING LINE. I AM NOT SURE WHAT THIS IS DOING. I DO NOT WANT THE NEIGHBORHOOD TO BE ERASED NOW.  
 // vertex.data().vid_set.clear(); //still necessary??    
  }

  //4-PROFILE CHANGE - now including a new scatter function for the 
  // equation involving H2,H7,H9 and H10.
  edge_dir_type scatter_edges(icontext_type& context,
                             const vertex_type& vertex) const {
    // return graphlab::OUT_EDGES; // we want scatter on the out-edges
    return graphlab::NO_EDGES;
  }
 
 // //need this for local only now????
  //  void scatter(icontext_type& context,
  //             const vertex_type& vertex,
  //             edge_type& edge) const {
  //   //    vertex_type othervtx = edge.target();
  //   if (edge.data().sample_indicator == 1){
  //     const vertex_data_type& srclist = edge.source().data();
  //     const vertex_data_type& targetlist = edge.target().data();
  //     // H2(CD) = |E|-x-1. x = Gamma(C)-1 + Gamma(D) -1 + n_{2,C}^{e}-n_{2,CD}^e +n_{2,D}^e-
  //     // n_{2,DC}^e+n_{3,C}+n_{3,D} -2*n_{3,CD} - H_7(CD)-H_9(CD)-H_10(CD)
  //     // Therefore, the constant is given by the RHS.
  //     //  H2(CD)-H_7(CD)-H9(CD)-H10(CD)= |E|-( Gamma(C)-1 + Gamma(D) -1 + n_{2,C}^{e}-n_{2,CD}^e +n_{2,D}^e-
  //     // n_{2,DE}^e+n_{3,C}+n_{3,D} -2*n_{3,CD} ) -1.C- src, D-target
  //     double x=0;
  //     x= srclist.vid_set.size() -1 +targetlist.vid_set.size()-1+srclist.num_wedges_e+targetlist.num_wedges_e - edge.data().n2e- edge.data().n2c+srclist.num_triangles+targetlist.num_triangles-2*edge.data().n3;
  //     edge.data().eqn10_const= total_edges - 1 -x;
  //     std::cout << "edge eqn10_const is: "<<edge.data().eqn10_const<<" "<<srclist.vid_set.size()<<" "<<targetlist.vid_set.size()<<" "<<srclist.num_wedges_e<<" "<<targetlist.num_wedges_e<<" "<<edge.data().n2e<<" "<<edge.data().n2c<<" "<<srclist.num_triangles<<"\n";

  //   }
  // } 

};


//probably have to change more stuff
//done after last previous scatter so these are final values at each vertex
//combine with clique counting??
/*class get_local_4prof :
      // public graphlab::ivertex_program<graph_type, size_t>,
      public graphlab::ivertex_program<graph_type, local4_gather>,
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
    local4_gather gather;
// 4 PROFILE CHANGES: ACTUALLY GATHERING FOR EQS (34) till (40) IN SCRATCH.
    if (edge.data().sample_indicator == 1){

      // std::cout << "edge.data().n2e = "<<edge.data().n2e<<" edge.data().n2c = "<<edge.data().n2c<<"\n";
      if (vertex.id() == edge.source().id()){
        gather.n3g = edge.target().data().num_triangles - edge.data().n3;
        gather.n2eg = edge.target().data().num_wedges_e - edge.data().n2c;
        gather.n0g = edge.target().data().num_empty;
      }
      else{
        gather.n3g = edge.source().data().num_triangles - edge.data().n3;
        gather.n2eg = edge.source().data().num_wedges_e - edge.data().n2c;
        gather.n0g = edge.source().data().num_empty;
      }
    }

    //stuff for clique counting?

    else{
      gather.n3g = 0;
      gather.n2eg = 0;
      gather.n0g = 0;
    }
    return gather;
  }
//apply
  void apply(icontext_type& context, vertex_type& vertex,
             const gather_type& ecounts) {
    vertex.data().u[13] = ecounts.n3g;
    vertex.data().u[14] = ecounts.n2eg;
    vertex.data().u[15] = ecounts.n0g;
    //stuff for clique counting?
  }
};*/


//IS THIS THE MOST RECENT CLIQUE COUNTING??
//now move clique counting from edge to vertex in gather
class eq10_count :
      public graphlab::ivertex_program<graph_type,
                                      h10_gather>,
      /* I have no data. Just force it to POD */
      public graphlab::IS_POD_TYPE  {
public:
//  bool do_not_scatter;

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
     h10_gather gather;
     if (edge.data().sample_indicator == 1){
        std::vector<graphlab::vertex_id_type> srcne = edge.source().data().vid_set.vid_vec;
        std::vector<graphlab::vertex_id_type> trgne = edge.target().data().vid_set.vid_vec;
        std::vector<graphlab::vertex_id_type> interlist;
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
        // //extra eqns
        // gather.n3g = edge.target().data().num_triangles - edge.data().n3;
        // gather.n2eg = edge.target().data().num_wedges_e - edge.data().n2c;
        // gather.n0g = edge.target().data().num_empty;     

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
        // //extra eqns
        // gather.n3g = edge.source().data().num_triangles - edge.data().n3;
        // gather.n2eg = edge.source().data().num_wedges_e - edge.data().n2c;
        // gather.n0g = edge.source().data().num_empty;

      }      
 
 
    }
    else {
      gather.common.clear();
      // gather.n3g = 0;
      // gather.n2eg = 0;
      // gather.n0g = 0;
    } 
    
    return gather;

  }

  /*
   * the gather result now contains the vertex IDs in the neighborhood.
   * store it on the vertex. 
   */
  void apply(icontext_type& context, vertex_type& vertex,
             const gather_type& ecounts) {
    if(ecounts.common.size()==0)
      vertex.data().conn_neighbors.clear();
    else   
      vertex.data().conn_neighbors=ecounts.common;
 // Just assigning the gathered list to just the appropriate vertex data
   //if(vertex.id()>=4039)
     //std::cout<<" vertex_id "<<vertex.id()<<":list of connected neighbor pairs "<<std::endl;  
    //for(size_t i=0;i<vertex.data().conn_neighbors.size();i++) {
    //if(vertex.data().conn_neighbors.at(i).first >= 4039 || vertex.data().conn_neighbors.at(i).second >= 4039)
    //std::cout<<vertex.data().conn_neighbors.at(i).first << " connected with "<<vertex.data().conn_neighbors.at(i).second<<std::endl;  }
  //}
    
    // //extra eqns
    // vertex.data().u[13] = ecounts.n3g;
    // vertex.data().u[14] = ecounts.n2eg;
    // vertex.data().u[15] = ecounts.n0g;
  }

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
      //const vertex_data_type& targetlist = edge.target().data();
       std::vector<graphlab::vertex_id_type> srcne = edge.source().data().vid_set.vid_vec;
       std::vector<graphlab::vertex_id_type>  trgne = edge.target().data().vid_set.vid_vec;
      std::vector<graphlab::vertex_id_type> interlist;
          interlist.clear();    
      sort(srcne.begin(),srcne.end());
      sort(trgne.begin(),trgne.end());
      edge.data().eqn10_const=0;     
  // interlist contains the intersection.  
      std::set_intersection(srcne.begin(),srcne.end(),trgne.begin(),trgne.end(),std::back_inserter(interlist));
      //Check for each pair of members if they have a common edge, if they do count.
   //   std::cout<<" src "<<edge.source().id()<< " target "<< edge.target().id()<<std::endl;
   //  std::cout<<" src "<<" connected neighbors";  

     //  for (size_t i=0;i<interlist.size();i++)
     //  std::cout<< interlist.at(i)<<std::endl;   

     //   for (size_t k=0;k<srclist.conn_neighbors.size();k++)
     //    std::cout<<srclist.conn_neighbors.at(k).first<<" with " <<srclist.conn_neighbors.at(k).second<<std::endl;
  
  
       for(size_t i=0;i<interlist.size();i++){
        for(size_t j=i+1;j<interlist.size();j++){
         
         for(size_t k=0;k<srclist.conn_neighbors.size();k++) {
            if ( ((srclist.conn_neighbors.at(k).first==interlist.at(i))&&(srclist.conn_neighbors.at(k).second==interlist.at(j)))||((srclist.conn_neighbors.at(k).first==interlist.at(j)) && (srclist.conn_neighbors.at(k).second==interlist.at(i))) )
            edge.data().eqn10_const++;
        //  std::cout<<srclist.conn_neighbors.at(k).first<<" with " <<srclist.conn_neighbors.at(k).second<<std::endl;
 
         }

        }
      }

  //ETHAN CHANGE
        //count every extra connected vertex exactly twice (scale by 12 now), but avoid a loop?
  //order dependent??!
  /*std::cout<<"edge "<<edge.source().id()<<" to "<<edge.target().id()<<std::endl;
  for (size_t ii=0; ii<interlist.size(); ii++) {
      for(size_t kk=0;kk<srclist.conn_neighbors.size();kk++) {
          std::cout<<srclist.conn_neighbors.at(kk).first<<" with " <<srclist.conn_neighbors.at(kk).second<<std::endl;
          if ( (srclist.conn_neighbors.at(kk).first==interlist.at(ii)) || (srclist.conn_neighbors.at(kk).second==interlist.at(ii)) )
              edge.data().eqn10_const++;
                                            
          if ( (srclist.conn_neighbors.at(kk).first==edge.target().id() && srclist.conn_neighbors.at(kk).second==interlist.at(ii)) || 
               (srclist.conn_neighbors.at(kk).second==edge.target().id() && srclist.conn_neighbors.at(kk).first==interlist.at(ii)) ) {
              std::cout<<" don''t count above"<<std::endl;
              edge.data().eqn10_const = edge.data().eqn10_const - 1;
         }
      }
        }
  edge.data().eqn10_const = (edge.data().eqn10_const)/2;*/

    //GET THIS AT THE EDGES
    // v.data().u[12] = ??v.data().eqn10_const??;

    }
 }     
};

class get_local_4prof :
      // public graphlab::ivertex_program<graph_type, size_t>,
      public graphlab::ivertex_program<graph_type, local4_gather>,
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
    local4_gather gather;
// 4 PROFILE CHANGES: ACTUALLY GATHERING FOR EQS (34) till (40) IN SCRATCH.
    if (edge.data().sample_indicator == 1){
      gather.h10v = edge.data().eqn10_const;

      // std::cout << "edge.data().n2e = "<<edge.data().n2e<<" edge.data().n2c = "<<edge.data().n2c<<"\n";
      if (vertex.id() == edge.source().id()){
        gather.n3g = edge.target().data().num_triangles - edge.data().n3;
        gather.n2eg = edge.target().data().num_wedges_e - edge.data().n2c;
        // gather.n0g = edge.target().data().num_empty;
        gather.n0g = edge.target().data().num_wedges_c - edge.data().n2c;
      }
      else{
        gather.n3g = edge.source().data().num_triangles - edge.data().n3;
        gather.n2eg = edge.source().data().num_wedges_e - edge.data().n2e;
        // gather.n0g = edge.source().data().num_empty;
        gather.n0g = edge.source().data().num_wedges_c - edge.data().n2e;
      }
    }

    //stuff for clique counting?

    else{
      gather.h10v = 0;
      gather.n3g = 0;
      gather.n2eg = 0;
      gather.n0g = 0;
    }
    return gather;
  }
//apply
  void apply(icontext_type& context, vertex_type& vertex,
             const gather_type& ecounts) {
    vertex.data().u[12] = ecounts.h10v;
    vertex.data().u[13] = ecounts.n3g;
    vertex.data().u[14] = ecounts.n2eg;
    vertex.data().u[15] = ecounts.n0g; //need another equation u[15] = ?????
    //NEED 1 MORE EQUATION!!!!!
    //stuff for clique counting?
  }

  edge_dir_type scatter_edges(icontext_type& context,
                             const vertex_type& vertex) const {
    return graphlab::NO_EDGES;
  }
};





//another engine, or just map? add to u[0] u[4] u[5] u[6], u[10]
//then invert matrix and store local 4-profile
//if not inverting (d-1 dimensional projection), then normalize the rows??

// vertex_data_type get_local_global_4profile(const graph_type::vertex_type& v) {
// uvec get_local_global_4profile(graph_type::vertex_type& v) {
void solve_local_4profile(graph_type::vertex_type& v) {
  // std::cout << "id: " << v.id() << ", global counts are "<< n[1] << ", " <<
  // n[2] << ", " << n[3] << std::endl;
  // std::cout << "id: " << v.id() << ", local counts are "<< v.data().num_disc << ", " <<
  // v.data().num_wedges << ", " << v.data().num_triangles << std::endl;
  /*
  std::cout << "id: " << v.id() << ", disconnected counts are "<< n[1] - v.data().num_disc << ", " <<
  n[2] - v.data().num_wedges << ", " << n[3] - v.data().num_triangles << std::endl;
  // std::cout << "last coeff: " << (total_vertices-1)*(total_vertices-2)*(total_vertices-3)/6 << std::endl;
  
  v.data().u[0] += n[1] - v.data().num_disc;
  v.data().u[4] += 2*(n[2] - v.data().num_wedges);
  v.data().u[5] += n[2] - v.data().num_wedges;
  v.data().u[6] += 2*(n[3] - v.data().num_triangles);
  v.data().u[10] = (total_vertices-1)*(total_vertices-2)*(total_vertices-3)/6; //|V|-1 choose 3
  */
  //reduce edges or gather or something to get num_edges at vertex??
  
  v.data().u[16] = (total_vertices-1)*(total_vertices-2)*(total_vertices-3)/6; //|V|-1 choose 3
  std::cout << "local u: ";
  for (int i=0; i< USIZE; i++){
    std::cout << v.data().u[i] << ", ";
  }
  std::cout << std::endl;
  // // //hardcode to debug, works so problem with the lhs constants or equations themselves
  // v.data().u[0] = 0;
  // v.data().u[1] = 0;
  // v.data().u[2] = 0;
  // v.data().u[3] = 3;
  // v.data().u[4] = 0;
  // v.data().u[5] = 0;
  // v.data().u[6] = 0;
  // v.data().u[7] = 0;
  // v.data().u[8] = 0;
  // v.data().u[9] = 0;
  // v.data().u[10] = 1;

  /* A*6 =
     -6     -2     -6     -2     -3     -6     -3     -3     -2     -3      0      0      0      0      0      0      6
      0      0    -10     -8     -2     -4      0      2      0     -4      4    -12     -1     10      4      2      0
      6      0     10      8      2      4      0     -2      0      4     -4     12      1    -10     -4     -2      0
      0      0      8      4      4      2      0     -4      0      2      4     12      2     -8     -2     -4      0
      0      0      2     -2      4     -1      0     -4      0     -1      1      6      2     -2      1     -1      0
      0      0     -8     -4     -4      4      0      4      0     -2     -4    -12     -2      8      2      4      0
      0      0     -4      4     -2      2      0      8      0      2     -2    -12     -4      4     -2      2      0
      0      0     -2      2     -1      1      3      1      0     -2     -1      0     -2      2     -1      1      0
      0      0      2      4     -2      2      0      2      0      2     -2      0     -1     -2     -2      2      0
      0      2      0      2      0      0      0      0     -1      0      0      0     -1      0      0      0      0
      0      0      2     -2      1     -1      0     -1      0     -1      1      6      2     -2      1     -1      0
      0      0      4     -4      2     -2      0     -2      0     -2      2      0      1      2      2     -2      0
      0      0      0     -6      0      0      0      0      3      0      0      0      3      0      0      0      0
      0      0      4     -4      2     -2      0     -2      0      4      2      0      4     -4      2     -2      0
      0      0     -2      2     -1      1      0      1      0      1     -1      0     -2      2     -1      1      0
      0      0      0      6      0      0      0      0      0      0      0      0     -3      0      0      0      0
      0      0      0      0      0      0      0      0      0      0      0      0      1      0      0      0      0
  
    const uvec A0  = {{-6,-2,-6,-2,-3,-6,-3,-3,-2,-3,0,0,0,0,0,0,6}};
    const uvec A1  = {{0,0,-10,-8,-2,-4,0,2,0,-4,4,-12,-1,10,4,2,0}};
    const uvec A2  = {{6,0,10,8,2,4,0,-2,0,4,-4,12,1,-10,-4,-2,0}};
    const uvec A3  = {{0,0,4,2,2,1,0,-2,0,1,2,6,1,-4,-1,-2,0}};
    const uvec A4  = {{0,0,2,-2,4,-1,0,-4,0,-1,1,6,2,-2,1,-1,0}};
    const uvec A5  = {{0,0,-4,-2,-2,2,0,2,0,-1,-2,-6,-1,4,1,2,0}};
    const uvec A6  = {{0,0,-2,2,-1,1,0,4,0,1,-1,-6,-2,2,-1,1,0}};
    const uvec A7  = {{0,0,-2,2,-1,1,3,1,0,-2,-1,0,-2,2,-1,1,0}};
    const uvec A8  = {{0,0,2,4,-2,2,0,2,0,2,-2,0,-1,-2,-2,2,0}};
    const uvec A9  = {{0,2,0,2,0,0,0,0,-1,0,0,0,-1,0,0,0,0}};
    const uvec A10 = {{0,0,2,-2,1,-1,0,-1,0,-1,1,6,2,-2,1,-1,0}};
    const uvec A11 = {{0,0,4,-4,2,-2,0,-2,0,-2,2,0,1,2,2,-2,0}};
    const uvec A12 = {{0,0,0,-2,0,0,0,0,1,0,0,0,1,0,0,0,0}};
    const uvec A13 = {{0,0,2,-2,1,-1,0,-1,0,2,1,0,2,-2,1,-1,0}};
    const uvec A14 = {{0,0,-2,2,-1,1,0,1,0,1,-1,0,-2,2,-1,1,0}};
    const uvec A15 = {{0,0,0,2,0,0,0,0,0,0,0,0,-1,0,0,0,0}};
    const uvec A16 = {{0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0}};
  */
      //NEED 1 MORE EQUATION!!!!!
      //???
      /*
      A*12 =
    -12     -4    -12     -4     -6    -12     -6     -6     -4     -6      0      0      0      0      0      0     12
      0      4    -18    -18      2    -14      0     -2     -2    -18     12     26     12    -20     -4      2      0
     12     -4     18     18     -2     14      0      2      2     18    -12    -26    -12     20      4     -2      0
      0     -8     12     12     -4     16      0      4      4     24      0    -28    -12     16      8     -4      0
      0     -2      3     -3      5      1      0     -5      1      3      0     -7      0     10      5     -1      0
      0      8    -12    -12      4     -4      0     -4     -4    -24      0     28     12    -16     -8      4      0
      0      4     -6      6      2     -2      0     10     -2     -6      0     14      0    -20    -10      2      0
      0      2     -3      3      1     -1      6     -1     -1     -9      0      7      0      2     -5      1      0
      0      4      6      6      2     -2      0     -2     -2     -6      0      2      0      4     -4      2      0
      0      4      0      4      0      0      0      0     -2      0      0      0      0      0     -2      0      0
      0     -2      3     -3     -1      1      0      1      1      3      0     -7      0     10      5     -1      0
      0     -4      6     -6     -2      2      0      2      2      6      0     -2      0     -4      4     -2      0
      0      0      0    -12      0      0      0      0      6      0      0      0      0      0      6      0      0
      0     -4      6     -6     -2      2      0      2      2     18      0    -14      0     -4     10     -2      0
      0      2     -3      3      1     -1      0     -1     -1     -3      0      7      0      2     -5      1      0
      0      0      0     12      0      0      0      0      0      0      0      0      0      0     -6      0      0
      0      0      0      0      0      0      0      0      0      0      0      0      0      0      2      0      0
    */

  
  const uvec A0  = {{-6,-2,-6,-2,-3,-6,-3,-3,-2,-3,0,0,0,0,0,0,6}};
  const uvec A1  = {{0,2,-9,-9,1,-7,0,-1,-1,-9,6,-10,-2,13,6,1,0}};
  const uvec A2  = {{6,-2,9,9,-1,7,0,1,1,9,-6,10,2,-13,-6,-1,0}};
  const uvec A3  = {{0,-2,3,3,-1,4,0,1,1,6,0,4,2,-7,-3,-1,0}};
  const uvec A4  = {{0,-2,3,-3,5,1,0,-5,1,3,0,10,5,-7,0,-1,0}};
  const uvec A5  = {{0,2,-3,-3,1,-1,0,-1,-1,-6,0,-4,-2,7,3,1,0}};
  const uvec A6  = {{0,2,-3,3,1,-1,0,5,-1,-3,0,-10,-5,7,0,1,0}};
  const uvec A7  = {{0,2,-3,3,1,-1,6,-1,-1,-9,0,2,-5,7,0,1,0}};
  const uvec A8  = {{0,2,3,3,1,-1,0,-1,-1,-3,0,2,-2,1,0,1,0}};
  const uvec A9  = {{0,2,0,2,0,0,0,0,-1,0,0,0,-1,0,0,0,0}};
  const uvec A10 = {{0,-2,3,-3,-1,1,0,1,1,3,0,10,5,-7,0,-1,0}};
  const uvec A11 = {{0,-2,3,-3,-1,1,0,1,1,3,0,-2,2,-1,0,-1,0}};
  const uvec A12 = {{0,0,0,-2,0,0,0,0,1,0,0,0,1,0,0,0,0}};
  const uvec A13 = {{0,-2,3,-3,-1,1,0,1,1,9,0,-2,5,-7,0,-1,0}};
  const uvec A14 = {{0,2,-3,3,1,-1,0,-1,-1,-3,0,2,-5,7,0,1,0}};
  const uvec A15 = {{0,0,0,2,0,0,0,0,0,0,0,0,-1,0,0,0,0}};
  const uvec A16 = {{0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0}};

  //write to vertex data in case saving local counts to file
  v.data().n4local[0] = (v.data().u * A0) / 6.;
  v.data().n4local[1] = (v.data().u * A1) / 6.;
  v.data().n4local[2] = (v.data().u * A2) / 6.;
  v.data().n4local[3] = (v.data().u * A3) / 3.;
  v.data().n4local[4] = (v.data().u * A4) / 12.;
  v.data().n4local[5] = (v.data().u * A5) / 3.;
  v.data().n4local[6] = (v.data().u * A6) / 6.;
  v.data().n4local[7] = (v.data().u * A7) / 12.;
  v.data().n4local[8] = (v.data().u * A8) / 6.;
  v.data().n4local[9] = (v.data().u * A9) / 6.;
  v.data().n4local[10] = (v.data().u * A10) / 12.;
  v.data().n4local[11] = (v.data().u * A11) / 6.;
  v.data().n4local[12] = (v.data().u * A12) / 2.;
  v.data().n4local[13] = (v.data().u * A13) / 6.;
  v.data().n4local[14] = (v.data().u * A14) / 12.;
  v.data().n4local[15] = (v.data().u * A15) / 2.;
  v.data().n4local[16] = (v.data().u * A16) / 6.;

  // return v.data().n4local;
  std::cout << "local 4-profile: ";
  for (int i=0; i< USIZE; i++){
    std::cout << v.data().n4local[i] << ", ";
  }
  std::cout << std::endl;
}

uvec get_global_4profile(const graph_type::vertex_type& v) {
  return v.data().n4local; //divide by 4 outside this function??
}




      


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

double get_eqn10_const(const graph_type::edge_type& e){
	return (double)(e.data().eqn10_const);
}

/*
 * A saver which saves a file where each line is a vid / # triangles pair
 */
struct save_profile_count{
  std::string save_vertex(graph_type::vertex_type v) { 
  //   double nt = v.data().num_triangles;
  //   double n_followed = v.num_out_edges();
  //   double n_following = v.num_in_edges();

  //   return graphlab::tostr(v.id()) + "\t" +
  //          graphlab::tostr(nt) + "\t" +
  //          graphlab::tostr(n_followed) + "\t" + 
  //          graphlab::tostr(n_following) + "\n";
  //
  /* WE SHOULD SCALE THE LOCAL COUNTS WITH p_sample BEFORE WRITING TO FILE!!!*/
  //just output u's instead since they will have the same classification power?
  //output local 3 profile also?
  std::string str = graphlab::tostr(v.id());
  for (int i=0; i<USIZE-1; i++){
    str += "\t" + graphlab::tostr(v.data().n4local[i]);
    // str += "\t" + graphlab::tostr(v.data().u[i]);
  }
  str += "\n";
  return str;
  // return graphlab::tostr(v.id()) + "\t" +
  //        graphlab::tostr(v.data().num_triangles) + "\t" +
  //        graphlab::tostr(v.data().num_wedges) + "\t" +
  //        graphlab::tostr(v.data().num_disc) + "\t" +
  //        graphlab::tostr(v.data().num_empty) + "\n";
  }
  std::string save_edge(graph_type::edge_type e) {
    return "";
  }
};


int main(int argc, char** argv) {

  graphlab::command_line_options clopts("4-profile Counting (redo this doc). "
    "Given an undirected graph, this program computes the frequencies of all "
    "subgraphs on 4 vertices (up to isomorphism). "
    "A file counts_4_profilesL.txt is appended with input filename, edge sampling probability, "
    "3-profile of the graph, 4-profile, and runtime. "
    "Network traffic is appended to netw_counts_4_profilesL.txt similarly. "
    "Option (per_vertex) not currently supported. "
    "The algorithm assumes that each undirected edge appears exactly once "
    "in the graph input. If edges may appear more than once, this procedure "
    "will over count. ");
  std::string prefix, format;
  std::string per_vertex;
  clopts.attach_option("graph", prefix,
                       "Graph input. reads all graphs matching prefix*");
  clopts.attach_option("format", format,
                       "The graph format");
 clopts.attach_option("ht", HASH_THRESHOLD,
                       "Above this size, hash sets are used");
  clopts.attach_option("per_vertex", per_vertex,
                       "Not currently supported. If not empty, will count the local "
                       "4-profile at each vertex and "
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

  total_vertices = graph.num_vertices();
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
    dc.cout() << "Counting 3-profiles..." << std::endl;
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
    //dc.cout() << "Round 2 Counted in " << ti2.current_time() << " seconds" << std::endl;
    //dc.cout() << "Total Running time is: " << ti.current_time() << "seconds" << std::endl;
    

    //if per vertex count, get cliques and calculate local count, else reduce and solve
    //check that reduced local and global are equal


    vertex_data_type global_counts = graph.map_reduce_vertices<vertex_data_type>(get_vertex_data);

    //global variables for the next transform
    n[0] = global_counts.num_empty/3;
    n[1] = global_counts.num_disc/3;
    n[2] = global_counts.num_wedges/3;
    n[3] = global_counts.num_triangles/3;
    
    dc.cout() << "Global count from estimators for 3-profiles: " 
        // << (global_counts.num_triangles/3)/pow(sample_prob_keep, 3) << " "
        // << (global_counts.num_wedges/3)/pow(sample_prob_keep, 2) - (1-sample_prob_keep)*(global_counts.num_triangles/3)/pow(sample_prob_keep, 3) << " "
       //  << (global_counts.num_disc/3)/sample_prob_keep - (1-sample_prob_keep)*(global_counts.num_wedges/3)/pow(sample_prob_keep, 2) << " "
        // << (global_counts.num_empty/3)-(1-sample_prob_keep)*(global_counts.num_disc/3)/sample_prob_keep  << " "
        // << std::endl;
        << (n[3])/pow(sample_prob_keep, 3) << " "
        << (n[2])/pow(sample_prob_keep, 2) - (1-sample_prob_keep)*(n[3])/pow(sample_prob_keep, 3) << " "
        << (n[1])/sample_prob_keep - (1-sample_prob_keep)*(n[2])/pow(sample_prob_keep, 2) << " "
        << (n[0])-(1-sample_prob_keep)*(n[1])/sample_prob_keep  << " "
        << std::endl;

    // ADDING 4-PROFILE STUFF HERE.
    //are these global, visible by get_local_global_4profile
    // const uvec A0 = {{-12, -7, 9, -4, -6, 0, -6, -6, -4, 0, 12}};
    // const uvec A1 = {{2, -1, 3, 0, -2, 4, 0, 0, 0, -2, 0}};
    // const uvec A2 = {{0, 1, -3, 0, 2, -4, 0, 0, 0, 2, 0}};
    // const uvec A3 = {{0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0}};
    // const uvec A4 = {{0, 0, 0, 0, -1, 2, 0, 0, 0, 0, 0}};
    // const uvec A5 = {{0, 3, -9, 0, 0, 0, 6, 0, 0, 0, 0}};
    // const uvec A6 = {{0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0}};
    // const uvec A7 = {{0, 0, 0, 0, 1, -2, 0, 1, 0, 0, 0}};
    // const uvec A8 = {{0, -1, 3, 0, 0, 0, 1, 0, 0, 24, 0}};
    // const uvec A9 = {{0, 1, -3, 0, 0, 0, 0, 0, 1, 0, 0}};
    // const uvec A10 = {{0, -1, 3, 2, 0, 0, 0, 0, -1, 0, 0}};

    // graph.transform_vertices(solve_local_4profile);
    
    //clique finding needed for both local and global
    graphlab::synchronous_engine<eq10_count> engine3(dc, graph, clopts);
    engine3.signal_all();
    engine3.start();  


    if (PER_VERTEX_COUNT == false) {
      // uvec n4final = graph.map_reduce_vertices<uvec>(get_global_4profile);
      // for (int i=0; i<USIZE; i++){
      //   n4final[i] = n4final[i]/4; //divide here?
      // }

      //map the global right hand sides to a vector for inverting
      uvec toInvert;
      toInvert[0] = global_counts.u[0];
      toInvert[1] = global_counts.u[1];
      toInvert[2] = global_counts.u[4];
      toInvert[3] = global_counts.u[6];
      toInvert[4] = global_counts.u[7];
      toInvert[5] = global_counts.u[8];
      toInvert[6] = global_counts.u[10];
      toInvert[7] = global_counts.u[3];
      toInvert[8] = global_counts.u[11];
      //eqn10 const only counts once per edge, scale differently for this system
      // toInvert[9] = graph.map_reduce_edges<double>(get_eqn10_const);
      toInvert[9] = graph.map_reduce_edges<double>(get_eqn10_const)*4.;

      // dc.cout() << "New global n_0a equation: " << global_counts.u[12] <<std::endl;
      

      double n4final[11] = {}; // This is for accumulating global 4-profiles.
      double denom = (graph.num_vertices()*(graph.num_vertices()-1)*(graph.num_vertices()-2)*(graph.num_vertices()-3))/24.; //normalize by |V| choose 4
      // global_counts.u[10] = denom; // the first 8 u's correspond to the 8 equations+ 9th equation is ethan's+ 10th equation needs to be assigned+11th is assigned (the one above) the sum of all counts. (V choose 4)
      toInvert[10] = denom; // the first 8 u's correspond to the 8 equations+ 9th equation is ethan's+ 10th equation needs to be assigned+11th is assigned (the one above) the sum of all counts. (V choose 4)
     
       // u [9] needs to be assigned the mapreduce of all eqn10_const variable from edge data.
      // global_counts.u[9] = graph.map_reduce_edges<double>(get_eqn10_const);
     
      // // Now the 11 rows of the inverse matrix is given below. - NEEDS TO BE CORRECTED !!!!
      // uvec A0 = {{-12, -8, -4, -12, -4, -6, -7, 6, 6, 24, 24}};
      // uvec A1 = {{2, 0, 2, 0, 0, 2, 1, -2, -4, -24, 0}};
      // uvec A2 = {{0, 0, -2, 0, 0, -2, -1, 2, 4, 24, 0}};
      // uvec A3 = {{0, 0, -2, 1, 0, -1, 0, 0, 2, 24, 0}};
      // uvec A4 = {{0, 0, 2, 0, 0, 1, 0, 0, -2, -24, 0}};
      // uvec A5 = {{0, 0, 2, 0, 1, 0, -1, 0, 0, -24, 0}};
      // uvec A6 = {{0, 2, 2, 0, 0, 0, -1, 0, 0, -24, 0}};
      // uvec A7 = {{0, 0, -1, 0, 0, 0, 0, 0, 1, 12, 0}};
      // uvec A8 = {{0, 0, -2, 0, 0, 0, 1, 0, 0, 24, 0}};
      // uvec A9 = {{0, 0, 1, 0, 0, 0, 0, 0, 0, -12, 0}};
      // // uvec A10 = {{0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0}};
     
      // // Operate Ai's on global_counts.u  
      // n4final[0] = (global_counts.u * A0) / 24.;
      // n4final[1] = (global_counts.u * A1) / 4.;
      // n4final[2] = (global_counts.u * A2) / 8.;
      // n4final[3] = (global_counts.u * A3) / 2.;
      // n4final[4] = (global_counts.u * A4) / 2.;
      // n4final[5] = (global_counts.u * A5) / 6.;
      // n4final[6] = (global_counts.u * A6) / 6.;
      // n4final[7] = (global_counts.u * A7) / 4.;
      // n4final[8] = (global_counts.u * A8) / 2.;
      // n4final[9] = (global_counts.u * A9) / 2.;
      // // n4final[10] = (global_counts.u * A10) / 1.;    
      // n4final[10] = global_counts.u[9];
      
      /*NEW A*24
     -6     -2     -6     -4     -6     -7     -6     -4      6      1     24
      0    -12    -12      0     12      6     12     12    -24     -6      0
      6      6      6      0     -6     -3     -6     -6     12      3      0
      0      0     12      0    -12      0      0    -24     24     12      0
      0      0      0      0     12      0      0     24    -24    -12      0
      0      0      0      4      0     -4      0      8      0     -4      0
      0      8      0      0      0     -4      0      8      0     -4      0
      0      0      0      0      0      0      0     -6      6      3      0
      0      0      0      0      0     12      0    -24      0     12      0
      0      0      0      0      0      0      0     12      0     -6      0
      0      0      0      0      0      0      0      0      0      1      0
      */

      uvec A0 = {{-6,-2,-6,-4,-6,-7,-6,-4,6,1,24}};
      uvec A1 = {{0,-2,-2,0,2,1,2,2,-4,-1,0}};
      uvec A2 = {{2,2,2,0,-2,-1,-2,-2,4,1,0}};
      uvec A3 = {{0,0,1,0,-1,0,0,-2,2,1,0}};
      uvec A4 = {{0,0,0,0,1,0,0,2,-2,-1,0}};
      uvec A5 = {{0,0,0,1,0,-1,0,2,0,-1,0}};
      uvec A6 = {{0,2,0,0,0,-1,0,2,0,-1,0}};
      uvec A7 = {{0,0,0,0,0,0,0,-2,2,1,0}};
      uvec A8 = {{0,0,0,0,0,1,0,-2,0,1,0}};
      uvec A9 = {{0,0,0,0,0,0,0,2,0,-1,0}};
      uvec A10 = {{0,0,0,0,0,0,0,0,0,1,0}};
     
      // Operate Ai's on global_counts.u  
      n4final[0] = (toInvert * A0) / 24.;
      n4final[1] = (toInvert * A1) / 4.;
      n4final[2] = (toInvert * A2) / 8.;
      n4final[3] = (toInvert * A3) / 2.;
      n4final[4] = (toInvert * A4) / 2.;
      n4final[5] = (toInvert * A5) / 6.;
      n4final[6] = (toInvert * A6) / 6.;
      n4final[7] = (toInvert * A7) / 8.;
      n4final[8] = (toInvert * A8) / 2.;
      n4final[9] = (toInvert * A9) / 4.;
      n4final[10] = (toInvert * A10) / 24.;    
      

      double n4finalEst[11] = {};
      //round or dont initialize as double?
      //probably should have done another inner product...
      /*n4finalEst[0] = round(n4final[0] - (1-sample_prob_keep)/sample_prob_keep*n4final[1]);
      n4finalEst[1] = round(n4final[1]/sample_prob_keep - (1-sample_prob_keep)/pow(sample_prob_keep,2)*(n4final[2] + n4final[3]));
      n4finalEst[2] = round(n4final[2]/pow(sample_prob_keep,2) - (1-sample_prob_keep)/(3*pow(sample_prob_keep,3))*n4final[4] - 
                        pow(1-sample_prob_keep,2)/(3*pow(sample_prob_keep,4))*n4final[7] - pow(1-sample_prob_keep,3)/(15*pow(sample_prob_keep,5))*n4final[9]);
      n4finalEst[3] = round(n4final[3]/pow(sample_prob_keep,2) - 2*(1-sample_prob_keep)/(3*pow(sample_prob_keep,3))*n4final[4] - 
                        (1-sample_prob_keep)/pow(sample_prob_keep,3)*(n4final[5]+n4final[6]) + 
                        pow(1-sample_prob_keep,2)/(3*pow(sample_prob_keep,4))*n4final[7] - pow(1-sample_prob_keep,3)/(15*pow(sample_prob_keep,5))*n4final[9]);
      n4finalEst[4] = round(n4final[4]/pow(sample_prob_keep,3) - (1-sample_prob_keep)/(2*pow(sample_prob_keep,4))*n4final[8] - 
                        pow(1-sample_prob_keep,2)/(5*pow(sample_prob_keep,5))*n4final[9]);
      n4finalEst[5] = round(n4final[5]/pow(sample_prob_keep,3) - (1-sample_prob_keep)/(4*pow(sample_prob_keep,4))*n4final[8]);
      n4finalEst[6] = round(n4final[6]/pow(sample_prob_keep,3) - (1-sample_prob_keep)/pow(sample_prob_keep,4)*n4final[7] - 
                        (1-sample_prob_keep)/(4*pow(sample_prob_keep,4))*n4final[8] + pow(1-sample_prob_keep,2)/(5*pow(sample_prob_keep,5))*n4final[9]);
      n4finalEst[7] = round(n4final[7]/pow(sample_prob_keep,4) - (1-sample_prob_keep)/(5*pow(sample_prob_keep,5))*n4final[9]);
      n4finalEst[8] = round(n4final[8]/pow(sample_prob_keep,4) - 4*(1-sample_prob_keep)/(5*pow(sample_prob_keep,5))*n4final[9]);
      n4finalEst[9] = round(n4final[9]/pow(sample_prob_keep,5) - (1-sample_prob_keep)/pow(sample_prob_keep,6)*n4final[10]);
      n4finalEst[10] = round(n4final[10]/pow(sample_prob_keep,6));*/

      //new estimator
      double tm = (sample_prob_keep-1)/sample_prob_keep;
      uvec E0 = {{1,tm,pow(tm,2),pow(tm,2),pow(tm,3),pow(tm,3),pow(tm,3),pow(tm,4),pow(tm,4),pow(tm,5),pow(tm,6)}};
      uvec E1 = {{0,1,2*tm,2*tm,3*pow(tm,2),3*pow(tm,2),4*pow(tm,3),4*pow(tm,3),5*pow(tm,4),6*pow(tm,5)}};
      uvec E2 = {{0,0,1,0,tm,0,0,2*pow(tm,2),pow(tm,2),2*pow(tm,3),3*pow(tm,4)}};
      uvec E3 = {{0,0,0,1,2*tm,3*tm,3*tm,4*pow(tm,2),5*pow(tm,2),8*pow(tm,3),12*pow(tm,4)}};
      uvec E4 = {{0,0,0,0,1,0,0,4*tm,2*tm,6*pow(tm,2),12*pow(tm,3)}};
      uvec E5 = {{0,0,0,0,0,1,0,0,tm,2*pow(tm,2),4*pow(tm,3)}};
      uvec E6 = {{0,0,0,0,0,0,1,0,tm,2*pow(tm,2),4*pow(tm,3)}};
      
      n4finalEst[0] = round(E0 * n4final);
      n4finalEst[1] = round((E1 * n4final)/sample_prob_keep);
      n4finalEst[2] = round((E2 * n4final)/pow(sample_prob_keep,2));
      n4finalEst[3] = round((E3 * n4final)/pow(sample_prob_keep,2));
      n4finalEst[4] = round((E4 * n4final)/pow(sample_prob_keep,3));
      n4finalEst[5] = round((E5 * n4final)/pow(sample_prob_keep,3));
      n4finalEst[6] = round((E6 * n4final)/pow(sample_prob_keep,3));
      n4finalEst[7] = round((n4final[7] + tm*n4final[9] + 3*pow(tm,2)*n4final[10])/pow(sample_prob_keep,4));
      n4finalEst[8] = round((n4final[8] + 4*tm*n4final[9] + 12*pow(tm,2)*n4final[10])/pow(sample_prob_keep,4));
      n4finalEst[9] = round((n4final[9] + 6*tm*n4final[10])/pow(sample_prob_keep,5));
      n4finalEst[10] = round(n4final[10]/pow(sample_prob_keep,6));
                  

      // DISPLAY STUFF.
      dc.cout() << "Global count of 4-profiles: "<<std::endl;
      for(int i=0; i<11; i++){
        dc.cout() << std::setprecision (std::numeric_limits<double>::digits10) << n4final[i] << " ";
      }
      dc.cout() << std::endl;
      dc.cout() << "Global count from estimators: "<<std::endl;
      for(int i=0; i<11; i++){
        dc.cout() << std::setprecision (std::numeric_limits<double>::digits10 + 3) << n4finalEst[i] << " ";
      }
      dc.cout() << std::endl;
      dc.cout() << "Global u: ";
      // for(int i=0; i<USIZE; i++){
      //   dc.cout() << global_counts.u[i] << " ";
      // }
      for(int i=0; i<11; i++){
        dc.cout() << toInvert[i] << " ";
      }
      dc.cout() << std::endl;

      total_time = ti.current_time();
      dc.cout() << "Total runtime: " << total_time << "sec." << std::endl;
      std::ofstream myfile;
      char fname[30];
      sprintf(fname,"counts_4_profilesL.txt");
      bool is_new_file = true;
      if (std::ifstream(fname)){
        is_new_file = false;
      }
      myfile.open (fname,std::fstream::in | std::fstream::out | std::fstream::app);
      if(is_new_file) myfile << "#graph\tsample_prob_keep\t3-profiles(4)\t4-profiles(11)\truntime" << std::endl;
      myfile << prefix << "\t"
             << sample_prob_keep << "\t"
             << std::setprecision (std::numeric_limits<double>::digits10 + 3)
             << round((n[3])/pow(sample_prob_keep, 3)) << "\t"
             << round((n[2])/pow(sample_prob_keep, 2) - (n[3])*(1-sample_prob_keep)/pow(sample_prob_keep, 3)) << "\t"
             << round((n[1])/sample_prob_keep - (n[2])*(1-sample_prob_keep)/pow(sample_prob_keep, 2)) << "\t"
             << round((n[0])-(n[1])*(1-sample_prob_keep)/sample_prob_keep)  << "\t";

      for(int i=0; i<11; i++)
        // myfile << n4final[i] <<"\t";
        myfile << n4finalEst[i] <<"\t";

      myfile  << std::setprecision (6)
             << total_time
             << std::endl;

      myfile.close();

      sprintf(fname,"netw_4_profL_%d.txt",dc.procid());
      myfile.open (fname,std::fstream::in | std::fstream::out | std::fstream::app);
      myfile << dc.network_bytes_sent() - reference_bytes <<"\n";
      myfile.close();



    }
    else {
      //compute additional vertex data for local count
      graphlab::synchronous_engine<get_local_4prof> engine4(dc, graph, clopts);
      engine4.signal_all();
      engine4.start();
      graph.transform_vertices(solve_local_4profile);
    
      //get global from local with global scaling
      // uvec n4final_l = graph.map_reduce_vertices<uvec>(get_global_4profile);
      vertex_data_type global_counts2 = graph.map_reduce_vertices<vertex_data_type>(get_vertex_data);

      double n4final_gfroml[11] = {};
      n4final_gfroml[0] = global_counts2.n4local[0]/4.;
      n4final_gfroml[1] = global_counts2.n4local[1]/2.;
      n4final_gfroml[2] = global_counts2.n4local[2]/4.;
      n4final_gfroml[3] = global_counts2.n4local[4];
      n4final_gfroml[4] = global_counts2.n4local[6]/2.;
      n4final_gfroml[5] = global_counts2.n4local[7]/3.;
      n4final_gfroml[6] = global_counts2.n4local[9];
      n4final_gfroml[7] = global_counts2.n4local[10]/4.;
      n4final_gfroml[8] = global_counts2.n4local[11];
      n4final_gfroml[9] = global_counts2.n4local[14]/2.;
      n4final_gfroml[10] = global_counts2.n4local[16]/4.;

      dc.cout() << "Global count from local 4-profiles: "<<std::endl;
      for(int i=0; i<11; i++){
        dc.cout() << std::setprecision (std::numeric_limits<double>::digits10 ) << n4final_gfroml[i] << " ";
      }
      dc.cout() <<std::endl;

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

