#include<iostream>
#include<cmath>
using namespace std;

int enumerate(int id){
	int sum=0;
	for(int i=id;i>0;i-=2){
		sum += 9*pow(10,i-1);
	}
	return sum;
}


int main(){
	string s;
	cin >> s;
	int a = atoi(s.c_str());
	int k = s.size();
	int l = a/pow(10,k-1);
	//cout << a << endl;
	//cout << k << endl;
	//cout << l << endl;
	int sum=0;
	if(k%2==0)  sum=enumerate(k-1);
	else{
		sum=enumerate(k-2);
		sum += (l-1)*pow(10,k-1);
		sum += a-l*pow(10,k-1)+1;
	}
	cout << sum << endl;
	return 0;
}



