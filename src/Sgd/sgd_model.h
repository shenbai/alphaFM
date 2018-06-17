#ifndef SGD_MODEL_H_
#define SGD_MODEL_H_

#include <unordered_map>
#include <string>
#include <vector>
#include <mutex>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include "../Utils/utils.h"

using namespace std;

//每一个特征维度的模型单元
class sgd_model_unit
{
public:
    double wi;
    vector<double> vi;
    double grad;
    int num;
    mutex mtx;
public:
    sgd_model_unit(int factor_num, double v_mean, double v_stdev)
    {
        wi = 0.0;
        vi.resize(factor_num);
        for(int f = 0; f < factor_num; ++f)
        {
            vi[f] = utils::gaussian(v_mean, v_stdev);
        }
        grad = 0.0;
        num = 0;
    }

    sgd_model_unit(int factor_num, const vector<string>& modelLineSeg)
    {
        vi.resize(factor_num);
        wi = stod(modelLineSeg[1]);
        for(int f = 0; f < factor_num; ++f)
        {
            vi[f] = stod(modelLineSeg[2 + f]);
        }
        grad = 0.0;
        num = 0;
    }

    void reinit_vi(double v_mean, double v_stdev)
    {
        int size = vi.size();
        for(int f = 0; f < size; ++f)
        {
            vi[f] = utils::gaussian(v_mean, v_stdev);
        }
    }

    void reset_grad()
    {
    	grad = 0.0;
    	num = 0;
    }

    friend inline ostream& operator <<(ostream& os, const sgd_model_unit& mu)
    {
        os << mu.wi;
        for(int f = 0; f < mu.vi.size(); ++f)
        {
            os << " " << mu.vi[f];
        }
        return os;
    }
};



class sgd_model
{
public:
    sgd_model_unit* muBias;
    vector<sgd_model_unit*> array;
    int factor_num;
    double init_stdev;
    double init_mean;
    int space_size;

public:
    sgd_model(double _factor_num, int _space_size);
    sgd_model(double _factor_num, int _space_size, double _mean, double _stdev);
    sgd_model_unit* getOrInitModelUnit(int index);
    sgd_model_unit* getOrInitModelUnitBias();

    double predict(const vector<pair<int, double> > &x, double bias, vector<sgd_model_unit *> &theta, vector<double> &sum);
    double getScore(const vector<pair<int, double> > &x, double bias);
    void outputModel(ofstream& out, bool compress);
    bool loadModel(ifstream& in);
    void debugPrintModel();

private:
    double get_wi(const int& index);
    double get_vif(const int& index, int f);
private:
    mutex mtx;
    mutex mtx_bias;
};

sgd_model::sgd_model(double _factor_num, int _space_size)
{
    space_size = _space_size;
    factor_num = _factor_num;
    init_mean = 0.0;
    init_stdev = 0.0;
    muBias = NULL;
    array.resize(space_size);
    cout << "space size: " << array.size() << endl;
}

sgd_model::sgd_model(double _factor_num, int _space_size, double _mean, double _stdev)
{
    space_size = _space_size;
    factor_num = _factor_num;
    init_mean = _mean;
    init_stdev = _stdev;
    muBias = NULL;
    array.resize(space_size);
}


sgd_model_unit* sgd_model::getOrInitModelUnit(int index)
{
    sgd_model_unit* p = array[index];
    if (p == NULL){
        mtx.lock();
        sgd_model_unit *pMU = new sgd_model_unit(factor_num, init_mean, init_stdev);
        array[index] = pMU;
        p = pMU;
        mtx.unlock();
    }
    return p;
}


sgd_model_unit* sgd_model::getOrInitModelUnitBias()
{
    if(NULL == muBias)
    {
        mtx_bias.lock();
        muBias = new sgd_model_unit(0, init_mean, init_stdev);
        mtx_bias.unlock();
    }
    return muBias;
}

double sgd_model::predict(const vector<pair<int, double> > &x, double bias, vector<sgd_model_unit *> &theta, vector<double> &sum)
{
    double result = 0;
    result += bias;
    for(int i = 0; i < x.size(); ++i)
    {
        result += theta[i]->wi * x[i].second;
    }
    double sum_sqr, d;
    for(int f = 0; f < factor_num; ++f)
    {
        sum[f] = sum_sqr = 0.0;
        for(int i = 0; i < x.size(); ++i)
        {
            d = theta[i]->vi[f] * x[i].second;
            sum[f] += d;
            sum_sqr += d * d;
        }
        result += 0.5 * (sum[f] * sum[f] - sum_sqr);
    }
    return result;
}

double sgd_model::getScore(const vector<pair<int, double> > &x, double bias)
{
    double result = 0;
    result += bias;
    for(int i = 0; i < x.size(); ++i)
    {
        result += get_wi(x[i].first) * x[i].second;
    }
    double sum, sum_sqr, d;
    for(int f = 0; f < factor_num; ++f)
    {
        sum = sum_sqr = 0.0;
        for(int i = 0; i < x.size(); ++i)
        {
            d = get_vif(x[i].first, f) * x[i].second;
            sum += d;
            sum_sqr += d * d;
        }
        result += 0.5 * (sum * sum - sum_sqr);
    }
    return 1.0/(1.0 + exp(-result));
}


double sgd_model::get_wi(const int& index)
{
    sgd_model_unit *p = array[index];
    if (p == NULL)
    {
        return 0.0;
    }
    else
    {
        return p->wi;
    }
}


double sgd_model::get_vif(const int& index, int f)
{
    sgd_model_unit *p = array[index];
    if (p == NULL)
    {
        return 0.0;
    }
    else
    {
        return p->vi[f];
    }
}


void sgd_model::outputModel(ofstream& out, bool compress)
{
    cout << utils::time_str() << " start dump model file" << endl;
    out << "bias " << *muBias << endl;
    int num = array.size();
    int i;
    int m = 0;
    stringstream ss;
    for(i = 0; i < num; i++){
      sgd_model_unit *p = array[i];
      if (p != NULL)
      {
        // w
        double d = 0.0;
        ss << i << " " << p->wi << " ";
        d += p->wi;
        // v
        for(int j=0; j< factor_num;j++)
        {
          ss << p->vi[j] << " ";
          d += p->vi[j];
        }
        if (p->wi == 0. and compress){
          ss.clear();
          ss.str("");
          continue;
        }
        ss << endl;
        if(m%100000==0)
        {
          // out << ss.str();
          string x = ss.str();
          out.write(x.c_str(), x.size());
          ss.clear();
          ss.str("");
          out.flush();
        }
        m++;
      }
    }
    // out << ss.str();
    string x = ss.str();
    out.write(x.c_str(), x.size());
    ss.clear();
    ss.str("");
    cout << utils::time_str() << " finish!" << endl;

}


void sgd_model::debugPrintModel()
{
    cout << "bias " << *muBias << endl;
    int num = array.size();
    int i;
    for(i = 0; i < num; i++){
      sgd_model_unit *p = array[i];
      cout << i << " " << p << endl;
    }
}


bool sgd_model::loadModel(ifstream& in)
{
    cout << utils::time_str() << " start load model ..." << endl;
    string line;
    if(!getline(in, line))
    {
      return false;
    }
    vector<string> strVec;
    utils::splitString(line, ' ', &strVec);
    if(strVec.size() != 1)
    {
        cout << "bias error" << endl;
        return false;
    }
    muBias = new sgd_model_unit(0, strVec);
    while(getline(in, line))
    {
        strVec.clear();
        utils::splitString(line, ' ', &strVec);
        if(strVec.size() != factor_num + 2)
        {
            cout << "vector size is " << strVec.size() << " except " << factor_num + 2 << endl;
            return false;
        }
        int index = stoi(strVec[0]);
        sgd_model_unit* pMU = new sgd_model_unit(factor_num, strVec);
        array[index] = pMU;
    }
    cout << utils::time_str() << "load model completed" << endl;
    return true;
}



#endif /*SGD_MODEL_H_*/
