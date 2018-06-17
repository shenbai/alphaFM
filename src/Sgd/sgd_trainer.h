#ifndef SGD_TRAINER_H_
#define SGD_TRAINER_H_
#include <iostream>
#include <fstream>
#include <stdio.h>  
#include "../Frame/pc_frame.h"
#include "../Sample/fm_sample.h"
#include "../Utils/utils.h"
#include <mutex>
#include "sgd_model.h"
#include "sgd_trainer_option.h"
#include <ctime>

using namespace std;

int sgd_line_num;
int sgd_train_num;
double sgd_train_loss;
int sgd_val_num;
double sgd_val_loss;

mutex sgd_mtx;

class sgd_trainer: public pc_task {
public:
	sgd_trainer(const sgd_trainer_option& opt);
	virtual void run_task(vector<string>& dataBuffer);
	bool loadModel(ifstream& in);
	void outputModel(ofstream& out);
	virtual ~sgd_trainer() {
	}
	;
private:
	void train(int y, const vector<pair<int, double> >& x);
private:
	sgd_model* pModel;
	float lr;
	double b_l1, b_l2;
	double w_l1, w_l2;
	double v_l1, v_l2;
	bool k0;
	bool k1;
	bool force_v_sparse;
	bool compress;
};

sgd_trainer::sgd_trainer(const sgd_trainer_option& opt) {
	lr = opt.lr;
	b_l1 = opt.b_l1;
	b_l2 = opt.b_l2;
	w_l1 = opt.w_l1;
	w_l2 = opt.w_l2;
	v_l1 = opt.v_l1;
	v_l2 = opt.v_l2;
	k0 = opt.k0;
	k1 = opt.k1;
	force_v_sparse = opt.force_v_sparse;
	compress = opt.compress;
	pModel = new sgd_model(opt.factor_num, opt.space_size, opt.init_mean,
			opt.init_stdev);
}

void sgd_trainer::run_task(vector<string>& dataBuffer) {
	for (int i = 0; i < dataBuffer.size(); ++i) {
		fm_sample sample(dataBuffer[i]);
		train(sample.y, sample.x);
	}
}

bool sgd_trainer::loadModel(ifstream& in) {
	return pModel->loadModel(in);
}

void sgd_trainer::outputModel(ofstream& out) {
	return pModel->outputModel(out, compress);
}

//void sgd_trainer::update(size_t offset, size_t len, double*& weight, double*& grad) {
//	for (size_t i = 0; i < len; i++) {
//		weight[i] -= __global_learning_rate * grad[i] / __global_minibatch_size;
//		grad[i] = 0.0f;
//	}
//}

//输入一个样本，更新参数
void sgd_trainer::train(int y, const vector<pair<int, double> >& x) {
	sgd_model_unit* thetaBias = pModel->getOrInitModelUnitBias();
	vector<sgd_model_unit*> theta(x.size(), NULL);
	int xLen = x.size();
	for (int i = 0; i < xLen; ++i) {
		const int& index = x[i].first;
		theta[i] = pModel->getOrInitModelUnit(index);
	}

	vector<double> sum(pModel->factor_num);
	double bias = thetaBias->wi;
	double p = pModel->predict(x, bias, theta, sum);
	double score = 1.0 / (1.0 + exp(-p));
	// train loss & val loss

	sgd_mtx.lock();
	sgd_line_num++;
	sgd_mtx.unlock();

	int _y = y > 0 ? 1 : 0;
	if (sgd_line_num % 1000 < 5) {
		sgd_mtx.lock();
		sgd_val_num++;
		sgd_val_loss += fabs(_y - score);
    cerr << _y << " " << score << endl;
		sgd_mtx.unlock();
	} else {
		sgd_mtx.lock();
		sgd_train_loss += fabs(_y - score);
		sgd_train_num++;
		sgd_mtx.unlock();

		if (sgd_train_num % 100000 == 0) {
      time_t now = time(0);
      tm *ltm = localtime(&now);
      cout << 1900 + ltm->tm_year << "-" << 1 + ltm->tm_mon
        << "-" << ltm->tm_mday << " " << ltm->tm_hour << ":"
        << ltm->tm_min << ":" << ltm->tm_sec << " ";
			cout << "line_num:" << sgd_line_num << "\ttrain_loss:"
					<< sgd_train_loss / sgd_train_num << "\tval_loss:"
					<< sgd_val_loss / sgd_val_num << "\tscore:" << score << endl;
      // cout << "score " << score << endl;
		}

    // mult = -train.target(train.data->getRowIndex())*(1.0-1.0/(1.0+exp(-train.target(train.data->getRowIndex())*p)));
    //
    double mult = -y * (1.0 - 1.0 / (1.0 + exp(-y * p)));
    sgd_mtx.lock();
    // cerr << "mult " << mult << " score " << score << " y " << _y << " p " << p << endl;
		sgd_mtx.unlock();

		//update w0
    //
		if (k0) {
			thetaBias->mtx.lock();
			double& w0 = thetaBias->wi;
			w0 -= 0.01 * lr * (mult + b_l1 * w0);
			thetaBias->mtx.unlock();
		}

		if (k1) {
			for (int i = 0; i < xLen; ++i) {
				sgd_model_unit& mu = i < xLen ? *(theta[i]) : *thetaBias;
				double xi = i < xLen ? x[i].second : 1.0;
				if ((i < xLen && k1)) {
					mu.mtx.lock();
					mu.wi -= lr
              // * (mult * xi + w_l1 * mu.wi + w_l2 * mu.wi * mu.wi);
              * (mult * xi  + w_l1 * mu.wi);
          // cout << "wi " << mu.wi << endl;
					mu.mtx.unlock();
				}
			}
		}

		for (int i = 0; i < xLen; ++i) {
			sgd_model_unit& mu = *(theta[i]);
			double xi = i < xLen ? x[i].second : 1.0;
			for (int f = 0; f < pModel->factor_num; ++f) {
				mu.mtx.lock();
				double& vif = mu.vi[f];

				if (force_v_sparse && 0.0 == mu.wi) {
					vif = 0.0;
				} else {
					double grad = 0;
					grad = sum[f] * xi - vif * xi * xi;
          // vif -= lr*0.01 * (mult * grad + v_l1 * vif + v_l2 * vif * vif);
          vif -= lr * (mult * grad);
				}
				mu.mtx.unlock();
			}
		}
	}

	//////////
	//pModel->debugPrintModel();
	//////////
}

#endif /*SGD_TRAINER_H_*/
