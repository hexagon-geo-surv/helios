#include "RisleyBeamDeflector.h"

#include <iostream>
#include <sstream>
using namespace std;

#define _USE_MATH_DEFINES
#include <math.h>
#include <logging.hpp>

#include <glm/glm.hpp>

#include "maths/Directions.h"
#include "MathConverter.h"

using Base = std::shared_ptr<AbstractBeamDeflector>;

// equations see: https://www.mdpi.com/1424-8220/21/14/4722

// equation (10)
namespace{
	glm::dvec3 calculate_refracted_ray(
		glm::dvec3 incident_ray,
		glm::dvec3 boundaryNormal,
		double ratio_of_refractive_indices)
	{
		double r = ratio_of_refractive_indices;
		double c = -glm::dot(incident_ray, boundaryNormal);
		return r * incident_ray + (r * c + sqrt(1 - r * r * (1 - c * c))) * boundaryNormal;
	}
}

// Construction/Cloning
Base RisleyBeamDeflector::clone(){
    Base ombd =
        std::make_shared<RisleyBeamDeflector>(
            RisleyBeamDeflector(
                cfg_device_scanAngleMax_rad,
                cfg_device_scanFreqMax_Hz,
                cfg_device_scanFreqMin_Hz
            )
        );
    _clone(ombd);
    return ombd;
}
void RisleyBeamDeflector::_clone(
    std::shared_ptr<AbstractBeamDeflector> abd
){
    AbstractBeamDeflector::_clone(abd);
    RisleyBeamDeflector *ombd = (RisleyBeamDeflector *)
        abd.get();
    ombd->scanAngle = scanAngle;
    ombd->rotorSpeed_rad_1 = rotorSpeed_rad_1;
    ombd->rotorSpeed_rad_2 = rotorSpeed_rad_2;
}

void RisleyBeamDeflector::applySettings(std::shared_ptr<ScannerSettings> settings) {
	AbstractBeamDeflector::applySettings(settings);
	cached_angleBetweenPulses_rad = (double)(this->cfg_setting_scanFreq_Hz * this->cfg_setting_scanAngle_rad * 4) / settings->pulseFreq_Hz;
	scanAngle = this->cfg_setting_scanAngle_rad;
	deltaT = 1.0 / settings->pulseFreq_Hz;
}

void RisleyBeamDeflector::doSimStep() {

	// time integration
	time += deltaT;

	// calculate the absolute angle
	
	double n_prism = 1.51;
	double n_air = 1.0;
	double prismAngle_rad = MathConverter::degreesToRadians(18.0);

	// check again (cf. https://www.mdpi.com/1424-8220/21/14/4722, but with
	// HELIOS++ reference coordinate system)
	// equations (3)-(6)

	glm::dvec3 l0(1.0, 0.0, 0.0);

	glm::dvec3 n2
	(
		cos(prismAngle_rad),
		-sin(rotorSpeed_rad_1 * time) * sin(prismAngle_rad),
		cos(rotorSpeed_rad_1 * time) * sin(prismAngle_rad)
	);

	glm::dvec3 n3
	(
		cos(prismAngle_rad),
		sin(rotorSpeed_rad_2 * time) * sin(prismAngle_rad),
		-cos(rotorSpeed_rad_2 * time) * sin(prismAngle_rad)
	);

	// first boundary passed: nothing happens as the ray is perprendicular to the boundary.

	// second boundary passed (from prism medium to air). Here the boundary is angled due to the prism shape.
	glm::dvec3 v2 = calculate_refracted_ray(l0, n2, n_prism / n_air);
	// third boundary passed (from air to prism medium). Here the boundary is angled due to the prism shape. 
	// Also the refraction of the previous boundary passing need to be taken into acoount.
	glm::dvec3 v3 = calculate_refracted_ray(v2, n3, n_air / n_prism);
	// fourth boundary passed (from prism medium to air). 
	// Here the boundary is again perpdendicular to the y axis but the previous refaction has to be taken into account.
	glm::dvec3 v4 = calculate_refracted_ray(v3, glm::dvec3(1.0, 0.0, 0.0), n_prism / n_air);

	// equations (11) and (12)
	// calculate yaw(phi) and pitch(eta) angles
	double phi = atan(v4.y / v4.x);
	double eta = acos(v4.z);

	// Rotate to current position:

	this->cached_emitterRelativeAttitude = Rotation(RotationOrder::ZYX, phi, eta, 0);
}

void RisleyBeamDeflector::setScanAngle_rad(double scanAngle_rad) {
	double scanAngle_deg = MathConverter::radiansToDegrees(scanAngle_rad);

	// Max. scan angle is limited by scan product:
	/*if (scanAngle_deg * this->cfg_setting_scanFreq_Hz > this->cfg_device_scanProduct) {
		logging::WARN(
		    "ERROR: Requested scan angle exceeds device limitations "
            "as defined by scan product. "
            "Will set it to maximal possible value."
        );
		scanAngle_deg = ((double) this->cfg_device_scanProduct) / this->cfg_setting_scanFreq_Hz;
	}*/

	this->cfg_setting_scanAngle_rad = MathConverter::degreesToRadians(scanAngle_deg);
	stringstream ss;
	ss << "Scan angle set to " << scanAngle_deg << " degrees.";
	logging::INFO(ss.str());
}

// This setter method should not be used for this scanner.

void RisleyBeamDeflector::setScanFreq_Hz(double scanFreq_Hz) {
	// Max. scan frequency is limited by scan product:
	//if( MathConverter::radiansToDegrees(this->cfg_setting_scanAngle_rad) *
	//    scanFreq_Hz > this->cfg_device_scanProduct
 //   ){
	//	logging::WARN(
	//	    "ERROR: Requested scan frequency exceeds device limitations "
 //           "as defined by scan product. "
 //           "Will set it to maximal possible value."
 //       );
	//	scanFreq_Hz = ((double) this->cfg_device_scanProduct) /
	//	    MathConverter::radiansToDegrees(this->cfg_setting_scanAngle_rad);
	//}
	this->cfg_setting_scanFreq_Hz = scanFreq_Hz;
	stringstream ss;
	ss << "Scan frequency set to " << this->cfg_setting_scanFreq_Hz << " Hz.";
	logging::INFO(ss.str());
}