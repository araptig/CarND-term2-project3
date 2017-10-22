/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[])
{// init
	// initialize random generators
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);
	num_particles = 200;

	Particle p;
	{// one particle, weights are all the same
		p.weight	= 1.0;
	}

	particles.resize(num_particles);
	for(int i=0; i<num_particles; i++)
	{
		p.id 	= i;
		p.x  	= dist_x(gen_);
		p.y  	= dist_y(gen_);
		p.theta = dist_theta(gen_);

		particles[i] = p;
	}

	weights.assign(num_particles,1.0);
	is_initialized = true;
	count=0;
	cout << "initialized" << endl << endl;
}// init

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate)
{//prediction
	// declare Gausiian RVs
	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_theta(0, std_pos[2]);

	// intermediate values, compute them once for all particles
	double delta_y = delta_t*yaw_rate;
	double delta_v =  delta_t*velocity;
	double ratio   = velocity/yaw_rate;

	for(int i=0; i<num_particles; i++)
	{//for each particle

		// (i) update position
		particles[i].x  	+= dist_x(gen_);				// obs noise
		particles[i].y  	+= dist_y(gen_);				// obs noise

		const double yaw          = particles[i].theta;
		const double cos_y 		=  cos(yaw);
		const double sin_y 		=  sin(yaw);
		if (yaw_rate < 0.002)
		{// assume yaw_rate =0
			particles[i].x  += delta_v*cos_y;
			particles[i].y  += delta_v*sin_y;
		}
		else
		{// yaw_rate != 0
		    double new_yaw = yaw + delta_y;
		    particles[i].x  +=  ratio * (sin(new_yaw) - sin_y);
		    particles[i].y  +=  ratio * (-cos(new_yaw) + cos_y);
		}

		{//(ii) update theta
			double theta         = particles[i].theta + dist_theta(gen_) + delta_y;
			//norm_angle(theta);
			particles[i].theta  = theta;					// predicted theta
		}
	}//for
}//prediction

void ParticleFilter::dataAssociation(const Map &map_landmarks, LandmarkObs & map_obs)
{//dataAssociation
	// Data association (VERY CONFUSING INTERAFCE!!!, changed arguments
	double distance = 1000000.0;
	int   id        = 0;
	for(int m=0; m<map_landmarks.landmark_list.size(); m++)
	{//landmarks
		const double lm_x = map_landmarks.landmark_list[m].x_f;
		const double lm_y = map_landmarks.landmark_list[m].y_f;
		double new_dist = dist(map_obs.x, map_obs.y, lm_x, lm_y);
		if (new_dist < distance)
		{
			distance = new_dist;
			id       = map_landmarks.landmark_list[m].id_i;
		}
	}//landmarks
	map_obs.id = id;   // use LandmarkObs.id for matching landscape
}//dataAssociation

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks)
{// updateWeights
	LandmarkObs map_obs;	//dummy observation

	weights.assign(num_particles,1);  // initialize weights
	double weight_sum = 0;

	// Gaussian pdf parameters
	const double c   = 1 / (2 * M_PI * std_landmark[0] * std_landmark[1]);	//constant
	const double v_x = 1 / (2 * std_landmark[0] * std_landmark[0]);			// x
	const double v_y = 1 / (2 * std_landmark[1] * std_landmark[1]);			// y


	for(int i=0; i<num_particles; i++)
	{//for each particle
		const double x_p = particles[i].x;
		const double y_p = particles[i].y;
		const double yaw = particles[i].theta;

		// update observation coordinates and ID
		for(int j=0; j<observations.size(); j++)
		{//for each observation
			// (a) prepare observations
			const double x_c = observations[j].x;	//car observations
			const double y_c = observations[j].y;	//car observations
			map_obs.x =  x_c*cos(yaw) - y_c*sin(yaw) +  x_p; // car --> map
			map_obs.y =  x_c*sin(yaw) + y_c*cos(yaw) +  y_p; // car --> map

			// (b) match observation to landmark
			dataAssociation(map_landmarks, map_obs);

			// (c) form weight
			double exponent_x2 = (map_obs.x - map_landmarks.landmark_list[map_obs.id-1].x_f);
			exponent_x2 *= exponent_x2;
			double exponent_y2 = (map_obs.y - map_landmarks.landmark_list[map_obs.id-1].y_f);
			exponent_y2 *= exponent_y2;
			weights[i] *= c*exp(-v_x*exponent_x2 - v_y*exponent_y2);
		}//obs
		weight_sum += weights[i];
	}//for each particle

	if (weight_sum > 0.0)
	{
		for(int i=0; i<num_particles; i++)
		{
			weights[i] /= weight_sum;
			particles[i].weight = weights[i];
		}

	}
	else
	{
		cout << "sum to zero" << endl;
		//exit(0);
	}

	count ++;	// for debugging
}// updateWeights

void ParticleFilter::resample()
{
	 discrete_distribution<> pdf_wt(weights.begin(), weights.end());

	 vector<Particle> chosen_particles;
	 chosen_particles.resize(num_particles);

	 for(int i = 0; i < num_particles; i++)
	 {//each particle
		 int index = pdf_wt(gen_);
		 chosen_particles[i] = particles[index];
	 }//each particle
	 particles = chosen_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations,
		std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{// used by main
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseX(Particle best)
{// used by main
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseY(Particle best)
{// used by main
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
