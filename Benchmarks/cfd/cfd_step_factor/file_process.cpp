#include <iostream>
#include <fstream>
#include <string>

using namespace std;

int main(){
	ifstream inFile("tmp.data");
	ofstream outFile("final.data");

	string line;
	string data;

	while(getline(inFile, line)){
		//cout << line << endl;
		data = line.substr(2);
		outFile << data << endl;
	}

	inFile.close();
	outFile.close();

	return 0;
}