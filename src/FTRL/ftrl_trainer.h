#ifndef FTRL_TRAINER_H_
#define FTRL_TRAINER_H_
#include <iostream>
#include <stdio.h>  
#include "../Frame/pc_frame.h"
#include "ftrl_model.h"
#include "../Sample/fm_sample.h"
#include "../Utils/utils.h"
#include <mutex>

using namespace std;
struct trainer_option
{
    trainer_option() : k0(true), k1(true), factor_num(8), init_mean(0.0), init_stdev(0.1), w_alpha(0.05), w_beta(1.0), w_l1(0.1), w_l2(5.0),
               v_alpha(0.05), v_beta(1.0), v_l1(0.1), v_l2(5.0), 
               threads_num(1), b_init(false), force_v_sparse(false),space_size(pow(2,28)), compress(false) {}
    string model_path, init_m_path;
    double init_mean, init_stdev;
    double w_alpha, w_beta, w_l1, w_l2;
    double v_alpha, v_beta, v_l1, v_l2;
    int threads_num, factor_num, space_size;
    bool k0, k1, b_init, force_v_sparse, compress;
    
    void parse_option(const vector<string>& args) 
    {
        int argc = args.size();
        if(0 == argc) throw invalid_argument("invalid command\n");
        for(int i = 0; i < argc; ++i)
        {
            if(args[i].compare("-m") == 0) 
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command -m\n");
                model_path = args[++i];
            }
            else if (args[i].compare("-s") == 0)
            {
                if (i == argc - 1)
                    throw invalid_argument("invalid command -s\n");
                space_size = stoi(args[++i]);
            }
            else if(args[i].compare("-dim") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command -dim\n");
                vector<string> strVec;
                string tmpStr = args[++i];
                utils::splitString(tmpStr, ',', &strVec);
                if(strVec.size() != 3)
                    throw invalid_argument("invalid command -dim ~\n");
                k0 = 0 == stoi(strVec[0]) ? false : true;
                k1 = 0 == stoi(strVec[1]) ? false : true;
                factor_num = stoi(strVec[2]);
            }
            else if(args[i].compare("-init_stdev") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command -init_stdev\n");
                init_stdev = stod(args[++i]);
            }
            else if(args[i].compare("-w_alpha") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command -w_alpha\n");
                w_alpha = stod(args[++i]);
            }
            else if(args[i].compare("-w_beta") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command -w_beta\n");
                w_beta = stod(args[++i]);
            }
            else if(args[i].compare("-w_l1") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command -w_l1\n");
                w_l1 = stod(args[++i]);
            }
            else if(args[i].compare("-w_l2") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command -w_l2\n");
                w_l2 = stod(args[++i]);
            }
            else if(args[i].compare("-v_alpha") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command -v_alpha\n");
                v_alpha = stod(args[++i]);
            }
            else if(args[i].compare("-v_beta") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command -v_beta\n");
                v_beta = stod(args[++i]);
            }
            else if(args[i].compare("-v_l1") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command -v_l1\n");
                v_l1 = stod(args[++i]);
            }
            else if(args[i].compare("-v_l2") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command -v_l1\n");
                v_l2 = stod(args[++i]);
            }
            else if(args[i].compare("-core") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command -core\n");
                threads_num = stoi(args[++i]);
            }
            else if(args[i].compare("-im") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command -im\n");
                init_m_path = args[++i];
                b_init = true; //if im field exits , that means b_init = true !
            }
            else if(args[i].compare("-fvs") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command -fvs\n");
                int fvs = stoi(args[++i]);
                force_v_sparse = (1 == fvs) ? true : false;
            }
            else if(args[i].compare("-compress") == 0)
            {
                if(i == argc - 1)
                    throw invalid_argument("invalid command -compress\n");
                int c = stoi(args[++i]);
                compress = (1 == compress) ? true : false;
            }
            else
            {
                throw invalid_argument("invalid command ~~\n");
                break;
            }
        }
    }

};


int num;
double total_loss;
mutex mtx;

class ftrl_trainer : public pc_task
{
public:
    ftrl_trainer(const trainer_option& opt);
    virtual void run_task(vector<string>& dataBuffer);
    bool loadModel(ifstream& in);
    void outputModel(ofstream& out);
private:
    void train(int y, const vector<pair<int, double> >& x);
private:
    ftrl_model* pModel;
    double w_alpha, w_beta, w_l1, w_l2;
    double v_alpha, v_beta, v_l1, v_l2;
    bool k0;
    bool k1;
    bool force_v_sparse;
    bool compress;
};


ftrl_trainer::ftrl_trainer(const trainer_option& opt)
{
    w_alpha = opt.w_alpha;
    w_beta = opt.w_beta;
    w_l1 = opt.w_l1;
    w_l2 = opt.w_l2;
    v_alpha = opt.v_alpha;
    v_beta = opt.v_beta;
    v_l1 = opt.v_l1;
    v_l2 = opt.v_l2;
    k0 = opt.k0;
    k1 = opt.k1;
    force_v_sparse = opt.force_v_sparse;
    compress = opt.compress;
    pModel = new ftrl_model(opt.factor_num, opt.space_size, opt.init_mean, opt.init_stdev);
}

void ftrl_trainer::run_task(vector<string>& dataBuffer)
{
    for(int i = 0; i < dataBuffer.size(); ++i)
    {
        fm_sample sample(dataBuffer[i]);
        train(sample.y, sample.x);
    }
}


bool ftrl_trainer::loadModel(ifstream& in)
{
    return pModel->loadModel(in);
}


void ftrl_trainer::outputModel(ofstream& out)
{
    return pModel->outputModel(out, compress);
}


//输入一个样本，更新参数
void ftrl_trainer::train(int y, const vector<pair<int, double> >& x)
{
    ftrl_model_unit* thetaBias = pModel->getOrInitModelUnitBias();
    vector<ftrl_model_unit*> theta(x.size(), NULL);
    int xLen = x.size();
    for(int i = 0; i < xLen; ++i)
    {
        const int& index = x[i].first;
        theta[i] = pModel->getOrInitModelUnit(index);
    }
    //update w via FTRL
    for(int i = 0; i <= xLen; ++i)
    {
        ftrl_model_unit& mu = i < xLen ? *(theta[i]) : *thetaBias;
        if((i < xLen && k1) || (i == xLen && k0))
        {
            mu.mtx.lock();
            if(fabs(mu.w_zi) <= w_l1)
            {
                mu.wi = 0.0;
            }
            else
            {
                if(force_v_sparse && mu.w_ni > 0 && 0.0 == mu.wi)
                {
                    mu.reinit_vi(pModel->init_mean, pModel->init_stdev);
                }
                mu.wi = (-1) *
                    (1 / (w_l2 + (w_beta + sqrt(mu.w_ni)) / w_alpha)) *
                    (mu.w_zi - utils::sgn(mu.w_zi) * w_l1);

            }
            mu.mtx.unlock();
        }
    }
    //update v via FTRL
    for(int i = 0; i < xLen; ++i)
    {
        ftrl_model_unit& mu = *(theta[i]);
        for(int f = 0; f < pModel->factor_num; ++f)
        {
            mu.mtx.lock();
            double& vif = mu.vi[f];
            double& v_nif = mu.v_ni[f];
            double& v_zif = mu.v_zi[f];
            if(v_nif > 0)
            {
                if(force_v_sparse && 0.0 == mu.wi)
                {
                    vif = 0.0;
                }
                else if(fabs(v_zif) <= v_l1)
                {
                    vif = 0.0;
                }
                else
                {
                    vif = (-1) *
                        (1 / (v_l2 + (v_beta + sqrt(v_nif)) / v_alpha)) *
                        (v_zif - utils::sgn(v_zif) * v_l1);
                }
            }
            mu.mtx.unlock();
        }
    }
    vector<double> sum(pModel->factor_num);
    double bias = thetaBias->wi;
    double p = pModel->predict(x, bias, theta, sum);
    double score = 1.0 / (1.0 + exp(-p));

    mtx.lock();
    // total_loss += y * log(score + 0.0000000001) + (1 - y)*log(1-score + 0.0000000001);
    int _y = y > 0 ? 1 : 0;
    total_loss += fabs(_y - score);
    num++;  
    if (num%1000000==0){
        cout << "loss :" << total_loss/num << "," << score << "," << _y << endl;
    }
    mtx.unlock();

    double mult = y * (1 / (1 + exp(-p * y)) - 1);

    //update w_n, w_z
    for(int i = 0; i <= xLen; ++i)
    {
        ftrl_model_unit& mu = i < xLen ? *(theta[i]) : *thetaBias;
        double xi = i < xLen ? x[i].second : 1.0;
        if((i < xLen && k1) || (i == xLen && k0))
        {
            mu.mtx.lock();
            double w_gi = mult * xi;
            double w_si = 1 / w_alpha * (sqrt(mu.w_ni + w_gi * w_gi) - sqrt(mu.w_ni));
            mu.w_zi += w_gi - w_si * mu.wi;
            mu.w_ni += w_gi * w_gi;
            mu.mtx.unlock();
        }
    }
    //update v_n, v_z
    for(int i = 0; i < xLen; ++i)
    {
        ftrl_model_unit& mu = *(theta[i]);
        const double& xi = x[i].second;
        for(int f = 0; f < pModel->factor_num; ++f)
        {
            mu.mtx.lock();
            double& vif = mu.vi[f];
            double& v_nif = mu.v_ni[f];
            double& v_zif = mu.v_zi[f];
            double v_gif = mult * (sum[f] * xi - vif * xi * xi);
            double v_sif = 1 / v_alpha * (sqrt(v_nif + v_gif * v_gif) - sqrt(v_nif));
            v_zif += v_gif - v_sif * vif;
            v_nif += v_gif * v_gif;
            //有的特征在整个训练集中只出现一次，这里还需要对vif做一次处理
            if(force_v_sparse && v_nif > 0 && 0.0 == mu.wi)
            {
                vif = 0.0;
            }
            mu.mtx.unlock();
        }
    }
    //////////
    //pModel->debugPrintModel();
    //////////
}


#endif /*FTRL_TRAINER_H_*/
