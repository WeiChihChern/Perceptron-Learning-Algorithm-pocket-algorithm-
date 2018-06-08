#include <iostream>
#include <fstream> 
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <ctime>

using namespace std;



void readTxt(string fileName, vector<vector<double>> &dst) {
	ifstream in(fileName);
	string line;
	while (getline(in, line)) {
		vector<double> dataEntry;
		stringstream s(line);
		for (int i = 0; i < 5; i++) {
			string item;
			if (i == 0)
				dataEntry.push_back(1.0); //x0 = 1;
			if (i != 3)
				getline(s, item, ' ');
			else
				getline(s, item, '\t');
			dataEntry.push_back(stod(item));
		}
		dst.push_back(dataEntry);
	}

}


//Perceptron Learning Algorithm
class PLA {
public:
	

	/* 
		Randomly choose training data
		Poket algorithm of PLA for non-linear separable cases
		Select data randomly 
	*/
	PLA(vector<vector<double>> &inputData, double learnRate, int updateNum, int ranBias = 0) {
		vector<int>     mistakeIdx;   // Store idx of training data that gave wrong result
		vector<int>     correct;
		int best = 0, c = 0;
		int dataNum = inputData.size();
		int l = inputData[0].size() - 1;  // Last element yn
		int inputDim = inputData[0].size() - 1;
		for (int i = 0; i < inputDim; i++) {
			weights.push_back(0);         //Starting with zeros as requried '0' 1 2 3 4 | 5
			pWeights.push_back(0);
		}
		vector<int> RandomSequences;
		for (int i = 0; i < inputData.size(); i++)
			RandomSequences.push_back(i);


		while (updateNum > 0) { // Compare how many decision boundaries
			//Generate random sequences
			srand(updateNum + ranBias);
			random_shuffle(RandomSequences.begin(), RandomSequences.end()); // Randomized sorted sequence
			c = 0;
			int i = 0;
			int d_num = dataNum;

			while (d_num) { 
				double result = PLA::dotProduct(weights, inputData[RandomSequences[i]]);

				// Store idx of the training data that PLA made mistake of
				if (PLA::sign(result) != inputData[RandomSequences[i]][l]) {
					mistakeIdx.push_back(RandomSequences[i]);
					i++; d_num--;
				}
				// No mistake, correct count + 1
				else {
					c++; i++; d_num--;  // Num of corrects, Next training data, How many counting left 
				}

				if (d_num == 0) {
					if (c > best) { 
						best = c; 
						PLA::vecCopy(weights, pWeights); }
					srand(updateNum + ranBias); random_shuffle(mistakeIdx.begin(), mistakeIdx.end());
					for (int j = 0; j < inputDim; j++)
						weights[j] += inputData[mistakeIdx[0]][j] * learnRate * inputData[mistakeIdx[0]][l];
				}
			}
			updateNum--;
		}



		PLA::vecCopy(pWeights, weights); 
	}






	double getErrorRate(vector<vector<double>> &testingInput) const{
		int s = testingInput.size();
		int l = testingInput[0].size()-1;
		double errCounts = 0;
		double result = 0;
		for (int i = 0; i < s; i++) {
			result = PLA::dotProduct(weights, testingInput[i]);
			if (PLA::sign(result) != testingInput[i][l]) {
				errCounts++;
			}
		}
		return (errCounts/s);
	}

	void printWeights() const {
		cout << "Weights: \n";
		for (int i = 0; i < weights.size(); i++)
			cout << weights[i] << '\n';
		cout << "\n";
	}

	static double dotProduct(vector<double> const &v1, vector<double> const &v2) {
		if (v1.size() + 1 != v2.size()) { cout << "dotProdcut error!";  return -1; };
		double sum = 0;
		int l = v1.size();
		for (int i = 0; i < l; i++)
			sum += v1[i] * v2[i];

		return sum;
	}

	static vector<int> randomSeqGenerator(int start, int end, int bias) {
		vector<int> seq;
		for (int i = 0; i < end; i++)
			seq.push_back(i);

		srand(bias);
		random_shuffle(seq.begin(), seq.end()); // Randomized sorted sequence
		return seq;
	}

	static int sign(double input) {
		return input > 0.0 ? 1 : -1;
	}

	static void vecCopy(vector<double> const &src, vector<double> &dst) {
		if (src.size() != dst.size()) { cout << "vecCopy error!\n\n"; return; }
		int l = src.size();
		for (int i = 0; i < l; i++)
			dst[i] = src[i];
	}

	int getUpdateCounts() {
		return counts;
	}



private:
	vector<double> weights;
	vector<double> pWeights; // Pockets weights
	int counts;
	double errorRate;

};







int main() {




	vector<vector<double>> testingInputs;
	vector<vector<double>> inputData2;
	readTxt("Data_18_test.txt", testingInputs);
	readTxt("Data_18_train.txt", inputData2);

	double errRate = 0;
	int numUpdate = 50;
	int numExperiements = 2000;
	int learningRate = 1;

	for (int i = 0; i < numExperiements; i++) {
		PLA pla3(inputData2, learningRate, numUpdate, i+time(0));
		double e = pla3.getErrorRate(testingInputs);
		errRate += e;
	}

	cout << "Ave error rate for" << numExperiements << " experiments is: " << (double)(errRate / numExperiements) << endl;

	
	system("pause");
	return 0;
}
