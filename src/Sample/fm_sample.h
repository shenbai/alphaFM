#ifndef FM_SAMPLE_H_
#define FM_SAMPLE_H_

#include <string>
#include <vector>

using namespace std;

const string spliter = " ";
const string innerSpliter = ":";


class fm_sample
{
public:
    float y;
    vector<pair<int, double> > x;
    fm_sample(const string& line);
};


fm_sample::fm_sample(const string& line) 
{
    this->x.clear();
    size_t posb = line.find_first_not_of(spliter, 0);
    size_t pose = line.find_first_of(spliter, posb);
    int label = atoi(line.substr(posb, pose-posb).c_str());
    this->y = label > 0 ? 1 : -1;
    if (label<=0){
      this->y = -1;
    } else if(label == 1){
      this->y = 1;
    } else if(label == 2){
      this->y = 1.2;
    } else if(label == 3){
      this->y = 1.5;
    } else if(label == 4){
      this->y = 2;
    } else{
      this->y = 1;
    }
    int key;
    double value;
    while(pose < line.size())
    {
        posb = line.find_first_not_of(spliter, pose);
        if(posb == string::npos)
        {
            break;
        }
        pose = line.find_first_of(innerSpliter, posb);
        if(pose == string::npos)
        {
            cout << "wrong line input\n" << line << endl;
            throw "wrong line input";
        }
        key = stoi(line.substr(posb, pose-posb));
        posb = pose + 1;
        if(posb >= line.size())
        {
            cout << "wrong line input\n" << line << endl;
            throw "wrong line input";
        }
        pose = line.find_first_of(spliter, posb);
        value = stod(line.substr(posb, pose-posb));
        if(value != 0)
        {
            this->x.push_back(make_pair(key, value));
        }
    }
}


#endif /*FM_SAMPLE_H_*/
