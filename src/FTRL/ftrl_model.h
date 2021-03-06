#ifndef FTRL_MODEL_H_
#define FTRL_MODEL_H_

#include <unordered_map>
#include <string>
#include <vector>
#include <mutex>
#include <iostream>
#include <sstream>
#include <cmath>
#include "../Utils/utils.h"

using namespace std;

//每一个特征维度的模型单元
class ftrl_model_unit
{
public:
    double wi;
    double w_ni;
    double w_zi;
    vector<double> vi;
    vector<double> v_ni;
    vector<double> v_zi;
    mutex mtx;
public:
    ftrl_model_unit(int factor_num, double v_mean, double v_stdev)
    {
        wi = 0.0;
        w_ni = 0.0;
        w_zi = 0.0;
        vi.resize(factor_num);
        v_ni.resize(factor_num);
        v_zi.resize(factor_num);
        for(int f = 0; f < factor_num; ++f)
        {
            vi[f] = utils::gaussian(v_mean, v_stdev);
            v_ni[f] = 0.0;
            v_zi[f] = 0.0;
        }
    }

    ftrl_model_unit(int factor_num, const vector<string>& modelLineSeg)
    {
        vi.resize(factor_num);
        v_ni.resize(factor_num);
        v_zi.resize(factor_num);
        wi = stod(modelLineSeg[1]);
        w_ni = stod(modelLineSeg[2 + factor_num]);
        w_zi = stod(modelLineSeg[3 + factor_num]);
        for(int f = 0; f < factor_num; ++f)
        {
            vi[f] = stod(modelLineSeg[2 + f]);
            v_ni[f] = stod(modelLineSeg[4 + factor_num + f]);
            v_zi[f] = stod(modelLineSeg[4 + 2 * factor_num + f]);
        }
    }

    void reinit_vi(double v_mean, double v_stdev)
    {
        int size = vi.size();
        for(int f = 0; f < size; ++f)
        {
            vi[f] = utils::gaussian(v_mean, v_stdev);
        }
    }

    friend inline ostream& operator <<(ostream& os, const ftrl_model_unit& mu)
    {
        os << mu.wi;
        for(int f = 0; f < mu.vi.size(); ++f)
        {
            os << " " << mu.vi[f];
        }
        os << " " << mu.w_ni << " " << mu.w_zi;
        for(int f = 0; f < mu.v_ni.size(); ++f)
        {
            os << " " << mu.v_ni[f];
        }
        for(int f = 0; f < mu.v_zi.size(); ++f)
        {
            os << " " << mu.v_zi[f];
        }
        return os;
    }
};



class ftrl_model
{
public:
    ftrl_model_unit* muBias;
    // unordered_map<string, ftrl_model_unit*> muMap;
    vector<ftrl_model_unit*> array;
    int factor_num;
    double init_stdev;
    double init_mean;
    int space_size;

public:
    ftrl_model(double _factor_num, int _space_size);
    ftrl_model(double _factor_num, int _space_size, double _mean, double _stdev);
    ftrl_model_unit* getOrInitModelUnit(int index);
    ftrl_model_unit* getOrInitModelUnitBias();

    double predict(const vector<pair<int, double> > &x, double bias, vector<ftrl_model_unit *> &theta, vector<double> &sum);
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

ftrl_model::ftrl_model(double _factor_num, int _space_size)
{
    space_size = _space_size;
    factor_num = _factor_num;
    init_mean = 0.0;
    init_stdev = 0.0;
    muBias = NULL;
    cout << "array size 1: " << array.size() << endl;
    array.resize(space_size);
    cout << "array size 2: " << array.size() << endl;
}

ftrl_model::ftrl_model(double _factor_num, int _space_size, double _mean, double _stdev)
{
    space_size = _space_size;
    factor_num = _factor_num;
    init_mean = _mean;
    init_stdev = _stdev;
    muBias = NULL;
    array.resize(space_size);
}


ftrl_model_unit* ftrl_model::getOrInitModelUnit(int index)
{
    ftrl_model_unit* p = array[index];
    if (p == NULL){
        mtx.lock();
        ftrl_model_unit *pMU = new ftrl_model_unit(factor_num, init_mean, init_stdev);
        array[index] = pMU;
        p = pMU;
        mtx.unlock();
    }
    return p;
}


ftrl_model_unit* ftrl_model::getOrInitModelUnitBias()
{
    if(NULL == muBias)
    {
        mtx_bias.lock();
        muBias = new ftrl_model_unit(0, init_mean, init_stdev);
        mtx_bias.unlock();
    }
    return muBias;
}

double ftrl_model::predict(const vector<pair<int, double> > &x, double bias, vector<ftrl_model_unit *> &theta, vector<double> &sum)
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

double ftrl_model::getScore(const vector<pair<int, double> > &x, double bias)
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


double ftrl_model::get_wi(const int& index)
{
    ftrl_model_unit *p = array[index];
    if (p == NULL)
    {
        return 0.0;
    }
    else
    {
        return p->wi;
    }
}


double ftrl_model::get_vif(const int& index, int f)
{
    ftrl_model_unit *p = array[index];
    if (p == NULL)
    {
        return 0.0;
    }
    else
    {
        return p->vi[f];
    }
}


void ftrl_model::outputModel(ofstream& out, bool compress)
{
    cout << utils::time_str() << " start dump model file" << endl;
    out << "bias " << *muBias << endl;
    int num = array.size();
    int i;
    int m = 0;
    stringstream ss;
    for(i = 0; i < num; i++){
      ftrl_model_unit *p = array[i];
      if (p != NULL)
      {
        // w
        //out << i << " " << p->wi << " ";
        double d = 0.0;
        ss << i << " " << p->wi << " ";
        d += p->wi;
        // v
        for(int j=0; j< factor_num;j++)
        {
          //out << p->vi[j] << " ";
          ss << p->vi[j] << " ";
          d += p->vi[j];
        }
        if (p->wi == 0 and compress){
          ss.clear();
          ss.str("");
          continue;
        }
        // wn wz
        //out << p->w_ni << " " << p->w_zi << " ";
        ss << p->w_ni << " " << p->w_zi << " ";
        // vn vz
        for(int j=0;j < factor_num; j++)
        {
          //out << p->v_ni[j] << " ";
          ss << p->v_ni[j] << " ";
        }
        for(int j=0;j< factor_num; j++)
        {
          //out << p->v_zi[j] << " ";
          ss << p->v_zi[j] << " ";
        }
        ss << endl;
        if(m%20000==0)
        {
          // out << ss.str();
          string x = ss.str();
          out.write(x.c_str(), x.size());
          ss.clear();
          ss.str("");
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


void ftrl_model::debugPrintModel()
{
    cout << "bias " << *muBias << endl;
    int num = array.size();
    int i;
    for(i = 0; i < num; i++){
      ftrl_model_unit *p = array[i];
      cout << i << " " << p << endl;
    }
    // for(unordered_map<string, ftrl_model_unit*>::iterator iter = muMap.begin(); iter != muMap.end(); ++iter)
    // {
      // cout << iter->first << " " << *(iter->second) << endl;
    // }
}


bool ftrl_model::loadModel(ifstream& in)
{
    cout << utils::time_str() << " start load model ..." << endl;
    string line;
    if(!getline(in, line))
    {
      return false;
    }
    vector<string> strVec;
    utils::splitString(line, ' ', &strVec);
    if(strVec.size() != 4)
    {
        cout << "bias error" << endl;
        return false;
    }
    muBias = new ftrl_model_unit(0, strVec);
    while(getline(in, line))
    {
        strVec.clear();
        utils::splitString(line, ' ', &strVec);
        if(strVec.size() != 3 * factor_num + 4)
        {
            cout << "vector size is " << strVec.size() << " except " << 3*factor_num + 4 << endl;
            return false;
        }
        int index = stoi(strVec[0]);
        ftrl_model_unit* pMU = new ftrl_model_unit(factor_num, strVec);
        // muMap[index] = pMU;
        array[index] = pMU;
    }
    cout << utils::time_str() << "load model completed" << endl;
    return true;
}



#endif /*FTRL_MODEL_H_*/
