#include "Static/BetweenCentrality/BetweenCentrality.cuh"
#include "Core/Auxilary/DuplicateRemoving.cuh"
#include <Graph/GraphStd.hpp>
#include <Graph/BC.hpp>

namespace hornets_nest
{

const dist_t INF = std::numeric_limits<dist_t>::max();

//------------------------------------------------------------------------------
///////////////
// OPERATORS //
///////////////

// O(max(queue.size)) -> worst case is when all edges go in one vertice
struct SSP
{
    dist_t *d_distances;
    dist_t current_level;
    float *sigma;
    TwoLevelQueue<vid_t> queue;

    OPERATOR(Vertex &vertex, Edge &edge)
    {
        auto dst = edge.dst_id();
        auto src = edge.src_id();

        if (atomicCAS(&d_distances[dst], INF,current_level+1) == INF)
        {   
            queue.insert(dst);
        }

        if (d_distances[dst] == d_distances[src] + 1)
        {
            atomicAdd(&sigma[dst], sigma[src]);
        }
    }
};

// O(max(queue.size)) -> the same of SSP
struct ACCUMULATE
{
    dist_t *d_distances;
    float *sigma;
    float *delta;
    OPERATOR(Vertex &vertex, Edge &edge)
    {
	        auto dst = edge.dst_id();
                auto src = edge.src_id();
                float add=(sigma[src] / sigma[dst]) * (1 + delta[dst]);
                if (d_distances[dst] == d_distances[src] + 1)
                {
                  atomicAdd(&delta[src], add);
                }

                
    }
};

// O(1) -> it's an array and each thread works on a single element
struct LOAD_S
{
    vid_t* S;
    int offset_length;
    OPERATOR(Vertex &vertex)
    {
	int tid = blockIdx.x*blockDim.x+threadIdx.x;	
        S[tid + offset_length] = vertex.id();
    }
};

// O(1) -> the same of LOAD_S, but queue.insert contains atomic operations?
struct LOAD_QUEUE
{
    int left_limit, right_limit;
    vid_t* S;
    TwoLevelQueue<vid_t> queue;
    OPERATOR(Vertex &vertex)
    {
	int i=blockIdx.x*blockDim.x+threadIdx.x;
        if (i <= right_limit && i >= left_limit)
                {
                    queue.insert(S[i]);
                }
    }
};

BetweenCentrality::BetweenCentrality(HornetGraph &hornet) : StaticAlgorithm(hornet),
                                                            queue(hornet),
                                                            load_balancing(hornet)
{
    gpu::allocate(d_distances, hornet.nV());
    gpu::allocate(sigma, hornet.nV());
    gpu::allocate(S, hornet.nV());
    gpu::allocate(delta, hornet.nV());
    gpu::allocate(bc, hornet.nV());
    reset();
}

void BetweenCentrality::run()	
{
    int offset_length = 0;
    vid_t tid;
  
	int num_nodes=hornet.nV();

    for (tid = 0; tid < hornet.nV(); tid++)	
    {

    int *ends = new int[hornet.nV() + 1];
        //INIT CONFIG
        ends[0] = 0;
        ends[1] = 1;
        int ends_l = 2;

        //CLEAR AND RESET
        this->reset();
        this->set_parameters(tid);


        current_level = 0;

        offset_length=0;
        //BFS
        while (queue.size() > 0)		
        { 
            forAllEdges(hornet, queue, SSP{d_distances, current_level, sigma, queue}, load_balancing);

            forAllVertices(hornet, queue, LOAD_S{S, offset_length});
            
            current_level++;
            offset_length += queue.size();

            queue.swap();
            ends[ends_l] = ends[ends_l - 1] + queue.size();
            
            ends_l++;
        }

        //ACCUMULATION STEP
        while (current_level > 0)		
        {   
            int left_limit = ends[current_level];
            int right_limit = ends[current_level + 1] - 1;
            forAllVertices(hornet, LOAD_QUEUE{left_limit, right_limit, S, queue});	
            queue.swap();          
            if(queue.size()>0) forAllEdges(hornet, queue, ACCUMULATE{d_distances, sigma, delta}, load_balancing); 
            current_level--;
        }

        auto delta_ = delta;
        auto bc_d=bc;
        forAllnumV(hornet, [=] __device__(int i) {bc_d[i]=bc_d[i]+delta_[i];});	
	delete[] ends;

    }
 }

BetweenCentrality::~BetweenCentrality()
{
    gpu::free(d_distances);
    gpu::free(delta);
    gpu::free(sigma);
    gpu::free(S);
    gpu::free(bc);
}

void BetweenCentrality::reset()
{
    current_level = 1;
    queue.clear();

    auto distances = d_distances;
    forAllnumV(hornet, [=] __device__(int i) { distances[i] = INF; });
    auto delta_ = delta;
    forAllnumV(hornet, [=] __device__(int i) { delta_[i] = 0; });
}

void BetweenCentrality::set_parameters(vid_t source)
{
    bfs_source = source;
    queue.insert(bfs_source);                  // insert bfs source in the frontier
    gpu::memsetZero(d_distances + bfs_source); //reset source distance
    auto distances = d_distances;
    forAllnumV(hornet, [=] __device__(int i) { distances[i] = INF; distances[source] = 0;});
    auto sigma_ = sigma;
    forAllnumV(hornet, [=] __device__(int i) {sigma_[i]=0;sigma_[source]=1; });
}

void BetweenCentrality::release()
{
    gpu::free(d_distances);
    gpu::free(sigma);
    gpu::free(S);
    gpu::free(delta);
    gpu::free(bc);
    d_distances = nullptr;
    S = nullptr;
    sigma = nullptr;
    delta = nullptr;
    bc=nullptr;
}
template<typename HostIterator, typename DeviceIterator>
bool equal(HostIterator host_start, HostIterator host_end,
           DeviceIterator device_start) noexcept {
    using R = typename std::iterator_traits<DeviceIterator>::value_type;
    auto size = std::distance(host_start, host_end);
    R* array = new R[size];
    cuMemcpyToHost(&(*device_start), size, array);

    bool flag;// = std::equal(host_start, host_end, array);
    flag=true;
    float epsilon=1;
    float maxErr=0;
    int index=0;
    if (true/*!flag*/) {
        for (int i = 0; i < size; i++) {
		float tmp=abs(host_start[i]-array[i]);
		if(maxErr<tmp){
			maxErr=tmp;
			index=i;
		}
            if (host_start[i]-array[i]>epsilon||host_start[i]-array[i]<-epsilon) {
                std::cout << std::setprecision(13) << host_start[i] << "  " << std::setprecision(13) << array[i] << "  at "
                          << i <<" with err value: "<<abs(host_start[i]-array[i])<< std::endl;
                flag=false;
            }
        }
    }
//    printf("MAX error precision: %.20f\t, on a total value of: %.10f\n", maxErr, host_start[index]);
    delete[] array;
    return flag;
}


bool BetweenCentrality::validate(timer::Timer<timer::DEVICE> TM)
{
    using namespace graph;
    GraphStd<vid_t, eoff_t> graph(hornet.csr_offsets(), hornet.nV(),
                                  hornet.csr_edges(), hornet.nE());
    BC<vid_t, eoff_t> betcen(graph);
    timer::Timer<timer::HOST> TM_H;
    TM_H.start();
  
    betcen.run(bfs_source);

    
    TM_H.stop();
    TM_H.print("Sequential: ");  
    printf("SpeedUP: %f\n",TM_H.duration()/TM.duration());
    auto sequentail_bc = betcen.result();
    return equal(sequentail_bc, sequentail_bc + graph.nV(), bc);
}
bool BetweenCentrality::validate()
{
    using namespace graph;
    GraphStd<vid_t, eoff_t> graph(hornet.csr_offsets(), hornet.nV(),
                                  hornet.csr_edges(), hornet.nE());
    BC<vid_t, eoff_t> betcen(graph);
    betcen.run(bfs_source);

    auto sequentail_bc = betcen.result();
    return gpu::equal(sequentail_bc, sequentail_bc + graph.nV(), bc);
}
} // namespace hornets_nest
