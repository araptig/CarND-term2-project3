/*
 * particle_filter.h
 *
 * 2D particle filter class.
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#ifndef PARTICLE_FILTER_H_
#define PARTICLE_FILTER_H_

#include <random>
#include "map.h"
#include "helper_functions.h"


struct Particle
{
	int id;
	double x;							// [m]
	double y;							// [m]
	double theta;						// orientation [rad]
	double weight;
	std::vector<int> associations;
	std::vector<double> sense_x;
	std::vector<double> sense_y;
};


class ParticleFilter
{//particle filter
private:
	bool is_initialized;				// Flag, if filter is initialized
	int num_particles; 					// Number of particles to draw
	std::vector<double> weights;		// Vector of weights of all particles
	std::default_random_engine gen_;          // init seed
	long count;
	
	// nearest neighbor landmark, put output in observations[].id
	void dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations);
	void dataAssociation(const Map &map_landmarks, LandmarkObs & observation);

public:
	std::vector<Particle> particles;	// Set of current particles

	ParticleFilter() : num_particles(0), is_initialized(false) {}
	~ParticleFilter() {}

	// initialize particle filter, std[3]
	void init(double x, double y, double theta, double std[]);

	//predict the next states using model, std_pos[3]
	void prediction(double delta_t, double std_pos[], double velocity, double yaw_rate);
	
	
	/**
	 * updateWeights Updates the weights for each particle based on the likelihood of the 
	 *   observed measurements. 
	 * @param sensor_range Range [m] of sensor
	 * @param std_landmark[] Array of dimension 2 [Landmark measurement uncertainty [x [m], y [m]]]
	 * @param observations Vector of landmark observations
	 * @param map Map class containing map landmarks
	 */
	void updateWeights(double sensor_range, double std_landmark[],
			const std::vector<LandmarkObs> &observations,
			const Map & map_landmarks);
	
	/**
	 * resample Resamples from the updated set of particles to form
	 *   the new set of particles.
	 */
	void resample();

	// Set a particles list of associations, along with the associations calculated world x,y coordinates
	// useful debugging tool to make sure transformations are correct and associations correctly connected
	//Particle SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x,
	//		std::vector<double> sense_y);
	Particle SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x,
			std::vector<double> sense_y);

	std::string getAssociations(Particle best);
	std::string getSenseX(Particle best);
	std::string getSenseY(Particle best);


	const bool initialized() const
	{// initialized Returns whether particle filter is initialized yet or not.
		return is_initialized;
	}
}; // particle filter

#endif
