#include<iostream>
using namespace std;

int main(){
    string str;
    cin >> str;
    int n = str.size();
    int count=0,color=0,max=0;
    int children[n],c[n];
    int childs[n];
    int base;
    for(int i=0;i<n;i++) childs[i]=0;
    for(int i=0;i<n;i++){
        
        if(str[i]=='R') c[i] =0;
        else c[i] =1;

        if(c[i] == color){
            count+=1; 
        }else{
            if(color==0){
                int _c = count/2;
                int c_ = count/2;
                if(count%2==1)c_ += 1;
                childs[i-1]+=c_;
                childs[i]+=_c;
                base = i;
                color=1;
            }else{
                int _c = count/2;
                int c_ = count/2;
                if(count%2==1)c_ += 1;
                childs[base] += c_;
                childs[base-1] += _c;
                //base = i+1;
                color=0;
            }
            count = 1;
            //color = c[i];
        } 
    }
    int _c = count/2;
                int c_ = count/2;
                if(count%2==1)c_ += 1;
                childs[base] += c_;
                childs[base-1] += _c;


    for(int i=0;i<n;i++) cout << childs[i] << " ";

    cout << endl;
    return 0;

}