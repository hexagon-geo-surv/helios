
#include <glm/gtx/norm.hpp>

#include "Survey.h"
#include <AbstractDetector.h>
#include <HeliosException.h>
#include <Simulation.h>
#include <SurveyPlayback.h>
#include <platform/InterpolatedMovingPlatformEgg.h>
#include <scene/StaticScene.h>
#include <scene/dynamic/DynScene.h>

namespace {
std::shared_ptr<Scene>
cloneScene(std::shared_ptr<Scene> const& scene)
{
  if (scene == nullptr)
    return nullptr;

  if (auto dynScene = std::dynamic_pointer_cast<DynScene>(scene)) {
    return std::make_shared<DynScene>(*dynScene);
  }
  if (auto staticScene = std::dynamic_pointer_cast<StaticScene>(scene)) {
    return std::make_shared<StaticScene>(*staticScene);
  }
  return std::make_shared<Scene>(*scene);
}
}

// ***  CONSTRUCTION / DESTRUCTION  *** //
// ************************************ //
Survey::Survey(Survey& survey, bool const deepCopy)
{
  // Copy basic attributes
  this->name = survey.name;
  this->numRuns = survey.numRuns;
  this->simSpeedFactor = survey.simSpeedFactor;
  this->length = survey.length;

  // Copy Scanner
  this->scanner = survey.scanner->clone();
  for (size_t i = 0; i < this->scanner->getNumDevices(); ++i) {
    this->scanner->getDetector(i)->scanner = this->scanner;
  }

  // Copy scene
  this->scene = deepCopy ? cloneScene(survey.getScene()) : survey.getScene();

  // Copy legs
  this->legs = std::vector<std::shared_ptr<Leg>>(0);
  for (size_t i = 0; i < survey.legs.size(); i++) {
    this->legs.push_back(std::make_shared<Leg>(*survey.legs[i]));
  }
}

// ***  M E T H O D S  *** //
// *********************** //
void
Survey::addLeg(int insertIndex, std::shared_ptr<Leg> leg)
{
  if (std::find(legs.begin(), legs.end(), leg) == legs.end()) {
    legs.insert(legs.begin() + insertIndex, leg);
  }
}

void
Survey::removeLeg(int legIndex)
{
  legs.erase(legs.begin() + legIndex);
}

std::shared_ptr<Scene>
Survey::getScene() const
{
  return scene;
}

Scene&
Survey::requireScene() const
{
  if (scene == nullptr) {
    throw HeliosException("Survey has no scene assigned");
  }
  return *scene;
}

void
Survey::setScene(std::shared_ptr<Scene> scene)
{
  this->scene = scene;
}

void
Survey::calculateLength()
{
  length = 0;
  for (size_t i = 0; i < legs.size() - 1; i++) {
    legs[i]->setLength(
      glm::distance(legs[i]->mPlatformSettings->getPosition(),
                    legs[i + 1]->mPlatformSettings->getPosition()));
    length += legs[i]->getLength();
  }
}

double
Survey::getLength()
{
  return this->length;
}

void
Survey::hatch(SurveyPlayback& sp)
{
  if (scanner->platform->isEgg()) {
    scanner->platform =
      std::static_pointer_cast<InterpolatedMovingPlatformEgg>(scanner->platform)
        ->smartHatch(sp.getStepLoop());
  }
}
