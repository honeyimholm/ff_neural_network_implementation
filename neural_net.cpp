/* In this program I will implement a neural network.
// The network consists of neurons, their associated connections, weights, and layer
// and the functions necessary to feedfoward, backpropagate and perform SGD
// Code for reading in the MNIST dataset was grabbed and modified from https://compvisionlab.wordpress.com/2014/01/01/c-code-for-reading-mnist-data-set/
// all other source code is mine.
*/ 
#include <iostream>
#include <math.h>
#include <random>
#include <iostream>
#include <vector>
#include <fstream>
using namespace std;
static int VARY_WEIGHT = 0;
static int VARY_BIAS = 1;

class Network {

public:
	int num_layers = 1;
	int size_in;
	int size_hidden;
	int size_out;
	int total_weights;
	int total_nodes;
	//NOTE: we number the weights going from top to bottom in each layer
	//the second weight going into third node of the third layer will be
	//weights[size_in*size_hidden+(3-1)*size_out+2]
	double*weights;
	double *biases;
	Network(int size_in, int size_hidden, int size_out);
	Network(int size_in, int size_hidden, int size_out, vector<double> init_weights, vector<double> init_biases);
	double **feedforward(double* input);
	double sigmoid(double input);
	double d_sigmoid(double input);
	void display_layer(int layer); //indexing at 0
	double** backpropagation(double *target, double *input, double *output, double learn_rate, double* hidden_out);
	void set_weights(double *new_weights);
	void set_biases(double *new_biases);
	void SGD(vector< vector<double> >training_input, vector< vector<double> >training_output, int data_num, int batch_size, double learning_rate, vector< vector<double> > testing_input, vector< vector<double> >testing_output, int test_data_num);
	double *eval(vector<vector<double>> testing_input, vector<vector<double>> testing_output, int data_num);
};

//Network constuctor with initial weights and biases
Network::Network(int size_in, int size_hidden, int size_out, vector<double> init_weights, vector<double> init_biases) {
	this->size_in = size_in;
	this->size_hidden = size_hidden;
	this->size_out = size_out;
	this->total_weights = size_in*size_hidden + size_hidden*size_out;
	this->total_nodes = size_in + size_hidden + size_out;
	this->weights = new double[total_weights];
	//intialize weights for the input/hidden layer with normalization of the square root of weights per node
	for (int i = 0; i < total_weights; i++) {
		weights[i]=init_weights[i];
	}
	//no biases for the input layer
	//need to randomly initialize the biases
	//NOTE: biases are assigned by layer
	this->biases = new double[total_nodes - size_in];
	for (int i = 0; i < total_nodes - size_in; i++) {
		biases[i] = init_biases[i];
	}
};

Network::Network(int size_in, int size_hidden, int size_out) {
	this->size_in = size_in;
	this->size_hidden = size_hidden;
	this->size_out = size_out;
	this->total_weights = size_in*size_hidden + size_hidden*size_out;
	this->total_nodes = size_in + size_hidden + size_out;
	this->weights = new double[total_weights];
	//need to randomly initialize the weights
	//std::random_device rd;  //Will be used to obtain a seed for the random number engine
	//std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
	std::default_random_engine gen;
	std::normal_distribution<double> dis(0, 1.0);
	std::normal_distribution<double> dis_b(0, 1.0);
	//TEST:
	//double test_weights[8] = { 1,1,1,-1,1,1,,.55 };
	//intialize weights for the input/hidden layer with normalization of the square root of weights per node
	for (int i = 0; i < total_weights; i++) {
		//input/hidden layer
		if (i < size_in*size_hidden) {
			weights[i] = dis(gen) / sqrt(double(size_in));
		}
		//hidden/output layer
		else
		{
			weights[i] = dis(gen)/sqrt(double(size_hidden));
		}
	}
	//no biases for the input layer
	//need to randomly initialize the biases
	//NOTE: biases are assigned by layer
	this->biases = new double[total_nodes - size_in];
	double temp = dis_b(gen);
	for (int i = 0; i < total_nodes - size_in; i++) {
		//UNDO:
		//new bias for every node
		if (i < size_hidden)
		{
			biases[i] = dis_b(gen) / sqrt(double(size_in));
		}
		biases[i] = dis_b(gen) / sqrt(double(size_hidden));
	}
};

//sigmoid function
double Network::sigmoid(double input) {
	return 1 / (1 + exp(-input));
}

//derivative of sigmoid
double Network::d_sigmoid(double input) {
	return sigmoid(input)*(1 - sigmoid(input));
}

//computes the activations for an input to the network
//returns an array containing the array's of the input, hidden and final output
//WARNING: assumes input is of correct size
double **Network::feedforward(double *input) {
	//calculate activations for input to hidden layer
	double *out_hidden = new double[size_hidden];
	for (int i = 0; i < size_hidden; i++) {
		out_hidden[i] = 0;
		for (int j = 0; j < size_in; j++) {
			//sum hidden node i's connected weight*input from input layer  
			out_hidden[i] += weights[j + i*size_in] * input[j];
		}
		out_hidden[i] = out_hidden[i] + biases[i];
		//update final activation for the nodes
		out_hidden[i] = sigmoid(out_hidden[i]);
	}
	//calculate activations for hidden layer to output layer
	//size_in->size_hidden, size_hidden->size_out, add offset to weight indexing
	//use out_hidden instead of input, add offset to biases indexing
	double *out_final = new double[size_out];
	for (int i = 0; i < size_out; i++) {
		out_final[i] = 0;
		for (int j = 0; j < size_hidden; j++) {
			//sum hidden node i's connected weight*input from input layer  
			out_final[i] += weights[j + size_in*size_hidden + i*size_hidden] * out_hidden[j];
		}
		//update final activation for the nodes
		out_final[i] = sigmoid(out_final[i] + biases[i + size_hidden]);
	}
	//constructing new memory for out_layers is necessary to retain pointer scope
	double **out_layers = new double*[3]; 
	out_layers[0] = input;
	out_layers[1] = out_hidden;
	out_layers[2] = out_final;
	return out_layers;
}

//helper function displays a layer's node biases and weights leading into it
void Network::display_layer(int layer) {
	switch (layer)
	{
	case 0:
		cout << "No weights or biases for the input layer..." << endl;
		break;
	case 1:
		//for (int i = 0; i < size_hidden; i++) {
		for (int i = 0; i < 1; i++) {
			cout << "Hidden Layer Node " << i << " bias: " << biases[i] << endl;
			for (int j = 0; j < size_in; j++) {
				//DEBUG
				if (i == 13 && j == 492) {
				//	system("pause");
				}
				cout << "\tWeight " << j << ": " << weights[j + i*size_in] << endl;
			}
		}
		break;
	case 2:
		for (int i = 0; i < size_out; i++) {
			cout << "Output Layer Node " << i << " bias: " << biases[i + size_hidden] << endl;
			for (int j = 0; j < size_hidden; j++) {
				cout << "\tWeight " << j << ": " << weights[j + size_in*size_hidden + i*size_hidden] << endl;
			}
		}
		break;
	default:
		cout << "Not a valid layer!" << endl;
		break;
	}
}

//WARNING: assumes target parameter is of correct size
double** Network::backpropagation(double *target, double*input, double *output, double learn_rate, double *hidden_out) {
	double *new_weights = new double[total_weights];
	double *new_biases = new double[total_nodes-size_in];
	double *error_mat = new double[size_out];
	//partial derivative of total error wrt output
	double *de_do = new double[size_out];
	//partial derivative of logistic function
	double *do_dn = new double[size_out];
	//partial derivative of net out wrt weight == node weight is connected to
	double *dn_dw = new double[size_out*size_hidden];
	double de_dw = 0; //temp var used for clarity
	for (int i = 0; i < size_out; i++) {
		error_mat[i] = pow((target[i] - output[i]), 2.0) / 2; //remove after debugging
		de_do[i] = -(target[i] - output[i]);
		do_dn[i] = output[i] * (1 - output[i]);
	}
	//NOTE: weights are reverse indexed by layer -> see note in class def
	//now we loop through the weights 2/3 layer to backrpopogate
	for (int i = 0; i < size_hidden*size_out; i++) {
		//which hidden node the weight is connected to
		dn_dw[i] = hidden_out[(i % size_out)];
		//putting it all together to find the new weight
		de_dw = de_do[i%size_out] * do_dn[i%size_out] * dn_dw[i];
		new_weights[i + size_hidden*size_in] = weights[i + size_hidden*size_in] - learn_rate*de_dw;
		//DEBUG
		//cout << "new weight " << i + size_hidden*size_in << ": " << new_weights[i + size_hidden*size_in] << endl;
		//DEBUG
		//cout << "weight " << i + size_hidden*size_in << " of node " << i%size_out << " is " << new_weights[i + size_hidden*size_in] << endl;
	}
	//TODO: update biases here
	//first implementation will change biases in place
	//TODO:return a temporary bias array to work with minibatching
	double bias_dn_dw = 1;
	for (int i = 0; i < size_out; i++)
	{
		//output of the bias node is always 1 
		//putting it all together to find the new weight
		de_dw = de_do[i%size_out] * do_dn[i%size_out] * bias_dn_dw;
		new_biases[i + size_hidden] = biases[i + size_hidden] - learn_rate*de_dw;
	}
	//partial derivative of error of output node wrt net of output node
	double deo_dno = 0;
	//partial derivative of total error wrt weight being solved for
	double det_dw = 0;
	//partial derivative of output of hidden node wrt network node
	double doh_dnh = 0;
	//partial derivative of net for hidden node wrt weight being solved for
	double dnh_dw = 0;
	//new
	double deo_doh = 0;
	double det_doh = 0; 
	//now we loop through hidden nodes
	for (int k = 0; k < size_hidden; k++) {
		//each hidden node loop through it's associated weights
		for (int w = 0; w < size_in; w++) {
			//each weight will be updated in this loop, sum for node delta calculated in loop below
			det_doh = 0;
			for (int j = 0; j < size_out; j++) {
				//uses previous results
				deo_dno = de_do[j] * do_dn[j]; 
				//index for weights finds weight connecting hidden node to output node 
				deo_doh = deo_dno*weights[size_hidden*size_in + j*size_hidden + k];
				det_doh += deo_doh;
			}
			doh_dnh = hidden_out[k] * (1 - hidden_out[k]); 
			//given input/hidden layer weight find corresponding input
			dnh_dw = input[w%size_in];
			//DEBUG
			if(input[w%size_in])
			{
				//cout << "DEBUG";
			}
			det_dw = doh_dnh*dnh_dw*det_doh;
			//implicit type conversion means we don't have to worry about the index expression
			//NOTE: indexing was initially wrong here! was using size_hidden*k for offset - be careful
			new_weights[w + size_in*k] = weights[w + size_in*k] - learn_rate*det_dw;
			//DEBUG
			//cout << "new weight " << w +size_in*k<< ": " << new_weights[w + size_in*k] << endl;
		}
	}
	//reinitialize for the bias calculation
	//partial derivative of error of output node wrt net of output node
	deo_dno = 0;
	//partial derivative of total error wrt weight being solved for
	det_dw = 0;
	//partial derivative of output of hidden node wrt network node
	doh_dnh = 0;
	//partial derivative of net for hidden node wrt weight being solved for
	dnh_dw = 0;
	//new
	deo_doh = 0;
	det_doh = 0;
	//set biases in place for the hidden layer
	for (int k = 0; k < size_hidden; k++)
	{
		det_doh = 0;
		for (int j = 0; j < size_out; j++) {
			//uses previous results
			deo_dno = de_do[j] * do_dn[j];
			//index for weights finds weight connecting hidden node to output node 
			//ERROR: j should be [0,size_out] NOT [0,size_hidden]
			deo_doh = deo_dno*weights[size_hidden*size_in + j*size_hidden + k];
			det_doh += deo_doh;
		}
		doh_dnh = hidden_out[k] * (1 - hidden_out[k]);
		//always 1 by definition of bias node
		dnh_dw = 1;
		det_dw = doh_dnh*dnh_dw*det_doh;
		//implicit type conversion means we don't have to worry about the index expression
		//NOTE: indexing was initially wrong here! was using size_hidden*k for offset - be careful
		new_biases[k] = biases[k] - learn_rate*det_dw;
	}
	//clear allocated memory to prevent leaks
	//keep new_weights because that's what we really care about!
	//NOTE: Make sure to delete new_weights in SGD when weights actually updated
	delete error_mat;
	delete dn_dw;
	delete de_do;
	delete do_dn;
	double **new_wb = new double*[2];
	new_wb[0] = new_weights;
	new_wb[1] = new_biases;
	return new_wb;
}

//sets the network weights to a different double ptr of weights
//WARNING: assumes new_weights are of consistent size!
//TODO: create error catching for inconsistent sizing
void Network::set_weights(double *new_weights) {
	//WARNING: complex memory dependencies introduced by this function
	//functionality is now to port over the weights - callee responsible for memory deallocation
	//delete this->weights;
	//this->weights = new_weights;
	for (int i = 0; i < total_weights ; i++) {
		//UNDO: used for testing one weight statically
		//if (i==VARY_WEIGHT)
		//{
		//DEBUG:
		//cout << "old weight: " << i << ": " << weights[i] << endl; 
			this->weights[i] = new_weights[i];
			//DEBUG:
			//cout << "new weight: " << i << ": " << weights[i] << endl;
		//}
	}
}

void Network::set_biases(double *new_biases) {
	//WARNING: complex memory dependencies introduced by this function
	//functionality is now to port over the weights - callee responsible for memory deallocation
	//delete this->weights;
	//this->weights = new_weights;
	for (int i = 0; i < total_nodes-size_in; i++) {
		//UNDO: commenting out for keeping network static except for one weight
		//if (i == VARY_BIAS){
			this->biases[i] = new_biases[i];
		//}
		
	}
}

//performs minibatching stochastic gradient descent
void Network::SGD(vector< vector<double> > training_input, vector< vector<double> >training_output, int data_num, int batch_size, double learning_rate, vector< vector<double> > testing_input, vector< vector<double> >testing_output, int test_data_num) {
	//net->display_layer(1);
	//net->display_layer(2);
	//will be used to store running sum of test weights
	double *batch_weights; 
	double *batch_biases;
	//one epoch loop
	//ofstream myfile;
	//UNDO: changed location of weight file
	//myfile.open("C:\\Users\\sahol\\Desktop\\Deep Learning\\SGD.txt");
	
	//looping for 1000 epochs 
	for (int e = 0; e < 10000; e++)
	{
		//evaluate the network every 200 epochs
		if(e%200==0){
			double *final_error = this->eval(testing_input, testing_output, test_data_num);
			//now run eval on the weights
			double sum = 0;
			for (int i = 0; i < size_out; i++)
			{
				sum += final_error[i];
			}
		}
		
		cout << "epoch" << e << endl;
		//UNDO
		//for (int b = 0; b < data_num / batch_size; b++)
		for (int b = 0; b < 1; b++)
		{
			//this->display_layer(2);
			//this->display_layer(1);
			//reallocate and free batch_weights every batch
			batch_weights = new double[total_weights];
			batch_biases = new double[total_nodes - size_in];
			//cout << "Entering minibatch " << b << "!!!"<<endl;
			//batch loop
			//initialize the batch_weights
			for (int j = 0; j < total_weights; j++) {
				batch_weights[j] = 0;
			}
			//initialize batch biases
			for (int j = 0; j < total_nodes-size_in; j++) {
				batch_biases[j] = 0;
			}
			int r = 0;
			for (int i = 0; i < batch_size; i++) {
				//WARNING: data is deallocated after each minibatch
				//we can convert vector to double pointer since addressing is garunteed contiguous
				//int ind = rand() % 4;
				r = rand() % data_num;
				cout << "RAND = " << r << endl;
				double *test_input = &training_input[r][0];//i + batch_size*b][0];
				//cout << "Test input: " << test_input[0]<<" "<<test_input[1]<<endl;
				double **ff = this->feedforward(test_input);
				double *test_output = ff[2];
				double *hidden_output = ff[1];
				double *t_input = ff[0];
				//REMOVE AFTER TESTIeNG
				//this->display_layer(1);
				//this->display_layer(2);
				//cout << "Input: "<< t_input[0] << " " << t_input[1] << endl << "Output: " << test_output[0] <<  endl;
				double *test_target = &training_output[r][0];//i + batch_size*b][0];
				//cout << "Target {01,.99}! Let's compute backpropagation: " << endl;
				double **bw = this->backpropagation(test_target, t_input, test_output, learning_rate, hidden_output);
				double *result = bw[0];
				double *new_biases = bw[1];
				//for (int i = 0; i < this->total_weights; i++) {
				//	cout << "\tWeight " << i + 1 << ":" << result[i] << endl;
				//}
				for (int j = 0; j < total_weights; j++) {
					batch_weights[j] = batch_weights[j] + result[j];
					//DEBUG
					//cout << "result weight " << j << ": " << result[j] << endl;
				}
				for (int j = 0; j < total_nodes-size_in; j++) {
					batch_biases[j] = batch_biases[j] + new_biases[j];
				}
				//clean up
				delete result;
				delete new_biases;
			}
			//divide all weights to get minibatch average
			for (int j = 0; j < total_weights; j++) {

				batch_weights[j] = batch_weights[j] / batch_size;
				//cout << "batch weight: " << batch_weights[j] << endl;
				//while setting the batch_weights we can also write out our updated weights to a file
				//UNDO: commenting out for testing static network
				//myfile << batch_weights[j] << " ";
			}
			//divide all biases to get minibatch average
			for (int j = 0; j < total_nodes-size_in; j++) {

				batch_biases[j] = batch_biases[j] / batch_size;
				//while setting the batch_weights we can also write out our updated weights to a file
				//UNDO:
				//myfile << batch_biases[j] << " ";
			}
			//UNDO:
			//myfile << batch_weights[VARY_WEIGHT] << " ";
			//myfile << batch_biases[VARY_BIAS] << " ";
			//myfile << sum / size_out;
			//TODO: smaller network can fit all weights on one line, will malfunction later
			//myfile << "\n";
			this->set_weights(batch_weights);
			this->set_biases(batch_biases);
			cout << "batch :" << b << endl;
			//double *final_error = this->eval(training_input, training_output, data_num);
			//for (int i = 0; i < this->size_out; i++) {
			//	cout << "Final Error Output Node " << i << ": " << final_error[i] << endl;
			//}
			//cout << "middle first: "<<weights[2]<<" last "<<weights[size_in*size_hidden+(size_out*size_hidden)-1]<<endl;
			//cout << "bias first: " << biases[0] << " last " << biases[size_out + size_hidden - 1] << endl;
			//this->display_layer(2);
			//this->display_layer(1);
			delete batch_weights;
			delete batch_biases;
		}
	}
	//myfile.close();
	//this->display_layer(2);
	//this->display_layer(1);
}
//helper function to evaluate the error of a network on a given test set
//outputs the average mean squared error over the test set
double* Network::eval(vector< vector<double> > testing_input, vector< vector<double> >testing_output, int data_num) {
		double *error_final = new double[size_out];
		for (int k = 0; k<size_out; k++) {
			error_final[k] = 0;
		}
		for (int i = 0; i < data_num; i++) {
		//WARNING: data is deallocated after each minibatch
		//we can convert vector to double pointer since addressing is garunteed contiguous
		double *test_input = testing_input[i].data();
		//cout << "Test input: " << test_input[0] << " " << test_input[1] << endl;
		double **ff = this->feedforward(test_input);
		double *test_output = ff[2];
		double *hidden_output = ff[1];
		double *t_input = ff[0];
		//cout << "Input: " << t_input[0] << " " << t_input[1] << endl << "Output: " << test_output[0] << " " << test_output[1] << " "<< test_output[2]<< test_output[3] << " " << test_output[4] << " " << test_output[5] << endl;
		double *test_target = testing_output[i].data();
		double error=0;
		//final error vector is the average of the mean squared error of the target and output
		//for MNIST the labels are values 0-9 and need to be formatted for error calculation 
		//double label_out = 0;
		//NOTE change line below if testing input changes
		//UNDO: uncomment the cout below
		//cout << "input: " << testing_input[i][0] << " " << testing_input[i][1] << endl;
		//cout << "\ttarget: " << test_target[0] << " output: " << test_output[0] <<endl;
		for (int j = 0; j < size_out; j++) {
			//should make error rate decrease
			//if (test_output[j] > .1) { test_output[j] = 1; }
			//else { test_output[j] = 0; }
			error = pow((test_target[j] - test_output[j]), 2.0) / 2;
			//if(i%(data_num/4)==0)
			cout << "\ttarget: " << test_target[j] << " output: " << test_output[j] <<endl;
			
			//cout << "\tData "<<i<<" Output node "<<j<<" Error: " << error << endl;
			error_final[j] += error;
		}
		//QUESTION: if a pixel input is always 0 does the weight ever change?
		//QUESTION: what will the error rate be if we take the maximum output node and set that to 
		//QUESTION: why is my output value capping around .5?
	}
	//divide cumulative error to get average mean sqaured error
	for(int k=0;k<size_out;k++)  {
		error_final[k] = error_final[k] / data_num;
		//cout << " Output node " << k  << " FINAL Error: " << error_final[k] << endl;
	}
	return error_final;
}

int ReverseInt(int i)
{
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}
void read_Mnist(string filename, vector<vector<double> > &vec, int data_num)
{
	ifstream file(filename, ios::binary);
	if (file.is_open())
	{
		int magic_number = 0x00000803;
		//placeholder value is replaced later
		int number_of_images = 1000;
		int n_rows = 28;
		int n_cols = 28;
		file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = ReverseInt(magic_number);
		file.read((char*)&number_of_images, sizeof(number_of_images));
		number_of_images = ReverseInt(number_of_images);
		file.read((char*)&n_rows, sizeof(n_rows));
		n_rows = ReverseInt(n_rows);
		file.read((char*)&n_cols, sizeof(n_cols));
		n_cols = ReverseInt(n_cols);
		number_of_images = data_num;
		for (int i = 0; i < number_of_images; ++i)
		{
			vector<double> tp;
			for (int r = 0; r < n_rows; ++r)
			{
				for (int c = 0; c < n_cols; ++c)
				{
					unsigned char temp = 0;
					file.read((char*)&temp, sizeof(temp));
					tp.push_back((double)temp);
				}
			}
			vec.push_back(tp);
		}
	}
}

void read_Mnist_Label(string filename, vector<double> &vec, int data_num)
{
	ifstream file(filename, ios::binary);
	if (file.is_open())
	{
		int magic_number = 0x00000801;
		int number_of_images = 1000; //is replaced later
		int n_rows = 0; //doesn't matter
		int n_cols = 0;
		file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = ReverseInt(magic_number);
		file.read((char*)&number_of_images, sizeof(number_of_images));
		number_of_images = ReverseInt(number_of_images);
		number_of_images = data_num;
		for (int i = 0; i < number_of_images; ++i)
		{
			unsigned char temp = 0;
			file.read((char*)&temp, sizeof(temp));
			vec[i] = (double)temp;
		}
	}
}
int main() {
	//net is dynamic 2x2x2 instance of network class we will use to test our functions
	//TODO: scale up the test for 100 input/output batch size 20
	int data_num = 10000;
	vector<vector<double>> ar_images;
	read_Mnist("C:\\train-images.idx3-ubyte", ar_images, data_num);
	//DEBUG
	//need to normalize the data to prevent vanishing gradient
	//when the input is above threshold map to 1 else 0
	int bw_threshold = 10;
	for (int i = 0; i < data_num; i++) {
		for(int j=0; j<784;j++)
		{	
			if (ar_images[i][j] > bw_threshold) {
				ar_images[i][j] = 1;
			}
			else {
				ar_images[i][j] = 0.0;
			}
		}
	
	}
	vector<double> ar_labels(data_num);
	read_Mnist_Label("C:\\train-labels.idx1-ubyte", ar_labels, data_num);
	vector<vector<double>> ar_labels_wrapper(data_num,vector<double>(1));
	
	for (int i = 0; i < data_num; i++){
		//UNDO: turned this into a binary classification problem
		//set up ar_labels_wrapper to transform label data from [0,9] to binary vector
		//for(int j=0;j<10;j++)
		//{
		//	ar_labels_wrapper[i][j] = double((int(ar_labels[i]) == j));//instead of creating switch case
		//}
		if (ar_labels[i] == 1)
		{
			ar_labels_wrapper[i][0] = 1;
		}
		else
		{
			ar_labels_wrapper[i][0] = 0;
		}
	}
	//now we will reduce the training set to only detect the digit one
	vector<vector<double>> ar_labels_wrapper_ones;
	vector<vector<double>> ar_images_ones;

	for (int i = 0; i < data_num; i++)
	{
		//if (ar_labels_wrapper[i][0] == 1)
		//{
		ar_labels_wrapper_ones.push_back(ar_labels_wrapper[i]);
		ar_images_ones.push_back(ar_images[i]);
		//}
	}
	data_num = ar_images_ones.size();
	//construct the testing set
	vector<vector<double>> ar_test_labels_ones;
	vector<vector<double>> ar_test_images_ones;
	int test_data_num = 40;
	for (int i = 0; i < test_data_num; i++)
	{
		ar_test_labels_ones.push_back(ar_labels_wrapper_ones[i]);
		ar_test_images_ones.push_back(ar_images_ones[i]);
	}
	//one epoch loop
	ofstream myfile;
	double *final_error;
	double error;
	Network *net = new Network(784, 100, 1);// , init_weights, init_biases);
	net->SGD(ar_images, ar_labels_wrapper, data_num, 1, 3, ar_test_images_ones, ar_test_labels_ones, test_data_num);
	//test on the same set of data
	double *final_error_2 = net->eval(ar_test_images_ones, ar_test_labels_ones, test_data_num);
	for (int i = 0; i < net->size_out; i++) {
		cout << "Final Error Output Node " << i << ": " << final_error_2[i]<<endl;
	}
	//net->display_layer(2);
	net->display_layer(1);
	system("pause");
	return 0;

}
