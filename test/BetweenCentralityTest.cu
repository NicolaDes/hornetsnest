/**
 * @brief Between centrality
 * @file
 */
#include "Static/BetweenCentrality/BetweenCentrality.cuh"
#include <Graph/GraphStd.hpp>
#include <Util/CommandLineParam.hpp>
#include <cuda_profiler_api.h> //--profile-from-start off

int main(int argc, char* argv[]) {
    using namespace timer;
    using namespace graph;
    using namespace hornets_nest;
    using namespace structure_prop;

    graph::GraphStd<vid_t, eoff_t> graph(ENABLE_INGOING);
    CommandLineParam cmd(graph, argc, argv);
    //graph.print();

    HornetInit hornet_init(graph.nV(), graph.nE(), graph.csr_out_offsets(),
                           graph.csr_out_edges());

    HornetGraph hornet_graph(hornet_init);

    BetweenCentrality betcen(hornet_graph);


    Timer<DEVICE> TM;
       cudaProfilerStart();
       TM.start();
   //    betcen.set_parameters(graph.max_out_degree_id());
       betcen.run();
   
       TM.stop();
       cudaProfilerStop();
       TM.print("Parallel: ");



	//graph.print();
    auto is_correct = betcen.validate(TM);

  std::cout << (is_correct ? "\nCorrect <>\n\n" : "\n! Not Correct\n\n");


return 0;
}
