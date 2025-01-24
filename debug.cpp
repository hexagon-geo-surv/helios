#include <sim/comps/Survey.h>
#include <python/PyXMLReader.h>
#include <chrono>
#include <SurveyPlayback.h>
// #include <SimulationCycleCallback.h>
// #include <SimulationCycleCallbackWrap.h>
#include <PulseThreadPoolFactory.h>

#include <filems/facade/FMSFacade.h>
#include <filems/facade/FMSFactoryFacade.h>
#include <filems/factory/FMSFacadeFactory.h>

bool    logging::LOGGING_SHOW_TRACE,    logging::LOGGING_SHOW_DEBUG,
        logging::LOGGING_SHOW_INFO,     logging::LOGGING_SHOW_TIME,
        logging::LOGGING_SHOW_WARN,     logging::LOGGING_SHOW_ERR;

using helios::filems::FMSFacadeFactory;

void test_run(std::shared_ptr<Survey> survey, size_t numThreads){

    survey->scanner->allOutputPaths =
            std::make_shared<std::vector<std::string>>(
                std::vector<std::string>(0)
            );
    survey->scanner->allMeasurements =
            std::make_shared<std::vector<Measurement>>(
                std::vector<Measurement>(0)
            );
    survey->scanner->allTrajectories =
            std::make_shared<std::vector<Trajectory>>(
                std::vector<Trajectory>(0)
            );
    survey->scanner->allMeasurementsMutex = std::make_shared<std::mutex>();

    std::shared_ptr<helios::filems::FMSFacade> fms = 
        helios::filems::FMSFacadeFactory().buildFacade(
            "./output",
            1.0,
            false,
            false,
            false,
            false,
            *survey
        );
    
    PulseThreadPoolFactory ptpf(
        1,
        numThreads-1,
        survey->scanner->getDetector()->cfg_device_accuracy_m,
        32,
        4
    );

    // Build pulse thread pool
    auto pulseThreadPool = ptpf.makePulseThreadPool();

    auto playback = std::make_shared<SurveyPlayback>(
        survey,
        fms,
        1,
        pulseThreadPool,
        32,
        "",
        true,
        true,
        true
    );
    playback->start();
}

int main(int argc, char *argv[]) {

    std::string surveyPath = "../data/surveys/toyblocks/als_toyblocks.xml";
    std::vector<std::string> assetsPath = {"../assets", "..", "../python/helios", "../python/helios/data"};
    bool legNoiseDisabled = true;
    bool rebuildScene = true;
    size_t numThreads = std::thread::hardware_concurrency();
    std::shared_ptr<Survey> survey = readSurveyFromXml(surveyPath, assetsPath, legNoiseDisabled, rebuildScene);
    
    for (int i = 0; i < std::stoi(argv[1]); i++) {
        auto start = std::chrono::high_resolution_clock::now(); 

        test_run(survey, numThreads);
        
        auto end = std::chrono::high_resolution_clock::now(); 
     
        std::chrono::duration<double> elapsed = end - start;
        
        std::cout << "Step " << i + 1 << " took " << elapsed.count() << " seconds." << std::endl;
    }

    return 0;
} 
