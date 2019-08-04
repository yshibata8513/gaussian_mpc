#include<iostream>
using namespace std;

int main(){
    int n;
    cin >> n;
    int h[n+1];
    for(int i=1;i<n+1;i++){
        cin >> h[i];
        if(i==1){
            h[i]-=1;
            continue;
        }
        if(h[i]>h[i-1]) h[i]-=1;
        if(h[i]<h[i-1]){
            cout << "No" << endl;
            return 0;
        }
    }
        
    cout << "Yes" << endl;

    return 0;
}