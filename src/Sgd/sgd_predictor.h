#ifndef SGD_PREDICTOR_H_
#define SGD_PREDICTOR_H_

#include "../Frame/pc_frame.h"
#include "../Sample/fm_sample.h"
#include "sgd_model.h"

class sgd_predictor: public pc_task {
public:
	sgd_predictor(double _factor_num, int _space_size, ifstream& _fModel,
			ofstream& _fPredict);
	virtual void run_task(vector<string>& dataBuffer);
	virtual ~sgd_predictor() {
	}
	;
private:
	sgd_model* pModel;
	ofstream& fPredict;
	mutex outMtx;
};

sgd_predictor::sgd_predictor(double _factor_num, int _space_size,
		ifstream& _fModel, ofstream& _fPredict) :
		fPredict(_fPredict) {
	pModel = new sgd_model(_factor_num, _space_size);
	if (!pModel->loadModel(_fModel)) {
		cout << "load model error!" << endl;
		exit(-1);
	}
}

void sgd_predictor::run_task(vector<string>& dataBuffer) {
	vector<string> outputVec(dataBuffer.size());
	for (int i = 0; i < dataBuffer.size(); ++i) {
		fm_sample sample(dataBuffer[i]);
		double score = pModel->getScore(sample.x, pModel->muBias->wi);
		outputVec[i] = to_string(sample.y) + " " + to_string(score);
	}
	outMtx.lock();
	for (int i = 0; i < outputVec.size(); ++i) {
		fPredict << outputVec[i] << endl;
	}
	outMtx.unlock();
}

#endif /*SGD_PREDICTOR_H_*/
