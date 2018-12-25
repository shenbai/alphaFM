#ifndef SGD_TRAINER_OPTION_H_
#define SGD_TRAINER_OPTION_H_
#include <iostream>
#include <stdio.h>  
#include "../Frame/pc_frame.h"
#include "sgd_model.h"
#include "../Sample/fm_sample.h"
#include "../Utils/utils.h"
#include <mutex>

using namespace std;
struct sgd_trainer_option {
	sgd_trainer_option() :
			k0(true), k1(true), factor_num(8), init_mean(0.0), init_stdev(0.1), w_l1(
					0.1), w_l2(5.0), v_l1(0.1), v_l2(5.0), b_l1(0.1), b_l2(0.1), threads_num(
					1), b_init(false), force_v_sparse(false), space_size(
					pow(2, 28)), compress(false), lr(0.01), batch(256) {
	}
	string model_path, init_m_path, opt;
	double init_mean, init_stdev, lr;
	double b_l1, b_l2;	// sgd
	double w_l1, w_l2;
	double v_l1, v_l2;
	int threads_num, factor_num, space_size, batch;
	bool k0, k1, b_init, force_v_sparse, compress;

	void parse_option(const vector<string>& args) {
		int argc = args.size();
		if (0 == argc)
			throw invalid_argument("invalid command\n");
		for (int i = 0; i < argc; ++i) {
			if (args[i].compare("-m") == 0) {
				if (i == argc - 1)
					throw invalid_argument("invalid command -m\n");
				model_path = args[++i];
			} else if (args[i].compare("-s") == 0) {
				if (i == argc - 1)
					throw invalid_argument("invalid command -s\n");
				space_size = stoi(args[++i]);
			} else if (args[i].compare("-opt") == 0) {
				if (i == argc - 1)
					throw invalid_argument("invalid command -opt\n");
				opt = args[++i];
			} else if (args[i].compare("-lr") == 0) {
				if (i == argc - 1)
					throw invalid_argument("invalid command -lr\n");
				lr = stod(args[++i]);
			} else if (args[i].compare("-dim") == 0) {
				if (i == argc - 1)
					throw invalid_argument("invalid command -dim\n");
				vector<string> strVec;
				string tmpStr = args[++i];
				utils::splitString(tmpStr, ',', &strVec);
				if (strVec.size() != 3)
					throw invalid_argument("invalid command -dim ~\n");
				k0 = 0 == stoi(strVec[0]) ? false : true;
				k1 = 0 == stoi(strVec[1]) ? false : true;
				factor_num = stoi(strVec[2]);
			} else if (args[i].compare("-init_stdev") == 0) {
				if (i == argc - 1)
					throw invalid_argument("invalid command -init_stdev\n");
				init_stdev = stod(args[++i]);
			} else if (args[i].compare("-b_l1") == 0) {
				if (i == argc - 1)
					throw invalid_argument("invalid command -b_l1\n");
				b_l1 = stod(args[++i]);
			} else if (args[i].compare("-b_l2") == 0) {
				if (i == argc - 1)
					throw invalid_argument("invalid command -b_l1\n");
				b_l1 = stod(args[++i]);
			} else if (args[i].compare("-w_l1") == 0) {
				if (i == argc - 1)
					throw invalid_argument("invalid command -w_l1\n");
				w_l1 = stod(args[++i]);
			} else if (args[i].compare("-w_l2") == 0) {
				if (i == argc - 1)
					throw invalid_argument("invalid command -w_l2\n");
				w_l2 = stod(args[++i]);
			} else if (args[i].compare("-v_l1") == 0) {
				if (i == argc - 1)
					throw invalid_argument("invalid command -v_l1\n");
				v_l1 = stod(args[++i]);
			} else if (args[i].compare("-v_l2") == 0) {
				if (i == argc - 1)
					throw invalid_argument("invalid command -v_l1\n");
				v_l2 = stod(args[++i]);
			} else if (args[i].compare("-core") == 0) {
				if (i == argc - 1)
					throw invalid_argument("invalid command -core\n");
				threads_num = stoi(args[++i]);
			} else if (args[i].compare("-im") == 0) {
				if (i == argc - 1)
					throw invalid_argument("invalid command -im\n");
				init_m_path = args[++i];
				b_init = true; //if im field exits , that means b_init = true !
			} else if (args[i].compare("-fvs") == 0) {
				if (i == argc - 1)
					throw invalid_argument("invalid command -fvs\n");
				int fvs = stoi(args[++i]);
				force_v_sparse = (1 == fvs) ? true : false;
			} else if (args[i].compare("-compress") == 0) {
				if (i == argc - 1)
					throw invalid_argument("invalid command -compress\n");
				int c = stoi(args[++i]);
				compress = (1 == compress) ? true : false;
			} else {
				throw invalid_argument("invalid command ~~\n");
				break;
			}
		}
	}

};


#endif /*SGD_TRAINER_OPTION_H_*/
