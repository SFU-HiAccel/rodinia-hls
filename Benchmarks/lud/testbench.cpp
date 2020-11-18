// #include "support.h"
#include <fstream>
#include <iostream>
#include "lud.h"
#include <iomanip>
#include <string>
#include <sstream>


using namespace std;
double GetFloatPrecision(double value, double precision)
{
    return (floor((value * pow(10, precision) + 0.5)) / pow(10, precision)); 
}


int main(){
	ifstream infile_input;
	ifstream infile_check;

	infile_input.open("input.data");
	infile_check.open("check.data");

	ofstream outfile_check;
	outfile_check.open("check_p.data");

	float result[65536];
	float reference[65536];
	int i = 0;

	float f;

	if(infile_input.is_open() && infile_check.is_open()){
		cout << "open success" << endl;
		for(int i = 0; i < 65536; i++){
			infile_input >> result[i];
			infile_check >> reference[i];
		}
	}

	workload(result);

	//Reference check

	int error = 0;

	for(int i = 0; i < 65536; i++){
		if(result[i] != reference[i]){
			float tmp;
			if(result[i] > reference[i])
				tmp = result[i] - reference[i];
			else
				tmp = reference[i] - result[i];

			float error_rate = tmp / result[i];

			if(error_rate > 0.005){
				error ++;
				if(error < 10){
					cout << i << endl;
					cout << setw(7) << result[i] << endl << setw(7) << reference[i] << endl;
				}
			}
			// float tmp_result = GetFloatPrecision(result[i], 3);
			// float tmp_reference = GetFloatPrecision(reference[i], 3);
			// if(tmp_result != tmp_reference){
			// 	error ++;
			// 	if(error < 10){
			// 		cout << i << endl;
			// 		cout << setw(7) << result[i] << endl << setw(7) << reference[i] << endl;
			// 	}
			// }
		}
	}

	cout << "error num:" << error << endl;

	return 0;
}