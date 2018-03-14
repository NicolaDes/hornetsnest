#pragma once

#include "HornetAlg.hpp"
//#include "Core/LoadBalancing/VertexBased.cuh"
//#include "Core/LoadBalancing/ScanBased.cuh"
#include "Core/LoadBalancing/BinarySearch.cuh"
#include <Core/GPUCsr/Csr.cuh>
#include <Core/GPUHornet/Hornet.cuh>
#include <Util/CommandLineParam.hpp>


namespace hornets_nest {

using HornetGraph = gpu::Csr<EMPTY, EMPTY>;

using dist_t = int;

class BetweenCentrality : public StaticAlgorithm<HornetGraph> {
public:
    BetweenCentrality(HornetGraph& hornet);
    ~BetweenCentrality();

    void reset()    override;
    void run()      override;
    void release()  override;
    bool validate() override;

    bool validate(timer::Timer<timer::DEVICE> TM);

    void set_parameters(vid_t source);
private:
    TwoLevelQueue<vid_t>        queue;
    load_balancing::BinarySearch load_balancing;
    // load_balancing::VertexBased1 load_balancing;
//    load_balacing::ScanBased load_balacing;
    dist_t* d_distances   { nullptr };
    dist_t* s_distances   {nullptr};
    vid_t   bfs_source    { 0 };
    dist_t  current_level { 0 };
    float* sigma {nullptr};
    dist_t* offset_depth {nullptr};
    vid_t* S {nullptr};
    float* delta {nullptr};
    float* bc {nullptr};
};

} // namespace hornets_nest
