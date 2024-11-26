#include <iostream>
#include <vector>
using namespace std;
int findMinimum(const std::vector<int>& arr) {
    if (arr.empty()) {
        throw invalid_argument("Array is empty");
    }
    
    int min = arr[0]; // Initialize min with the first element
    
    for (size_t i = 1; i < arr.size(); ++i) { // Start from the second element
        if (arr[i] < min) { // Compare and update min
            min = arr[i];
        }
    }
    
    return min; // Return the smallest value
}

int main() {
    vector<int> arr = {7, 3, 1, 8, 4, 2}; 
    
    try {
        int minimum = findMinimum(arr);
        cout << "The minimum value in the array is: " << minimum << endl;
    } catch (const exception& e) {
        cerr << e.what() << endl;
    }
    
    return 0;
}
