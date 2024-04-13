#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <cstring>
#include <mpi.h>
#include <map>
#include <chrono>

using namespace std;
using namespace std::chrono;

const int MAX_CHARS = 20;

// Define a structure to hold traffic data
struct Data {
    char time[MAX_CHARS]; // Timestamp of the traffic data
    int lightId; // ID of the traffic light
    int carsCount; // Number of cars at the traffic light
};

// Comparator function for sorting traffic data based on timestamp
bool sortData(const Data& x, const Data& y) {
    // Convert timestamps to strings for comparison
    string xTime(x.time);
    string yTime(y.time);
    // Sort first by timestamp, then by number of cars
    if (xTime == yTime) {
        return x.carsCount < y.carsCount;
    }
    return xTime < yTime;
}

int main(int argc, char** argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    auto start = high_resolution_clock::now(); // Start timing

    MPI_Datatype MPI_DATA_TYPE;
    MPI_Datatype MPI_STRING_TYPE;

    // Define MPI datatype for the custom structure
    MPI_Type_contiguous(MAX_CHARS, MPI_CHAR, &MPI_STRING_TYPE);
    MPI_Type_commit(&MPI_STRING_TYPE);
    int blockLengths[] = {MAX_CHARS, 1, 1};
    MPI_Aint displacements[3];
    MPI_Datatype types[] = {MPI_CHAR, MPI_INT, MPI_INT};
    Data temp;
    MPI_Get_address(&(temp.time), &displacements[0]);
    MPI_Get_address(&(temp.lightId), &displacements[1]);
    MPI_Get_address(&(temp.carsCount), &displacements[2]);
    displacements[1] -= displacements[0];
    displacements[2] -= displacements[0];
    displacements[0] = 0;
    MPI_Type_create_struct(3, blockLengths, displacements, types, &MPI_DATA_TYPE);
    MPI_Type_commit(&MPI_DATA_TYPE);

    if (rank == 0) {
        string filename = "traffic_data.txt";
        ifstream file(filename);
        if (!file) {
            cerr << "Error opening file" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        vector<Data> dataList;
        string line;
        getline(file, line); // Skip header line
        // Read data from file and populate the vector
        while (getline(file, line)) {
            istringstream iss(line);
            string ind, time, lightIdStr, carsCountStr;
            getline(iss, ind, ',');
            getline(iss, time, ',');
            getline(iss, lightIdStr, ',');
            getline(iss, carsCountStr, ',');
            Data data;
            strncpy(data.time, time.c_str(), MAX_CHARS - 1);
            data.time[MAX_CHARS - 1] = '\0';
            data.lightId = stoi(lightIdStr);
            data.carsCount = stoi(carsCountStr);
            dataList.push_back(data);
        }
        file.close();

        // Get timestamp range from user
        string startTimestamp, endTimestamp;

        cout << "\n";
        cout << "<---------- Welcome To Traffic Controller Simulator ----------> ";
        cout << "\n";
        cout << "\n";
        cout << "Enter the starting time (HH:MM:SS): ";
        cin >> startTimestamp;
        cout << "Enter the ending time (HH:MM:SS): ";
        cin >> endTimestamp;
        cout << "\n";
        cout << "<------------------ Processing The Data -----------------> ";
        cout << "\n";
        cout << "\n";
        cout << endl; // New line for better formatting

        vector<Data> filteredData;
        // Filter data based on user-provided start and end times
        for (const auto& data : dataList) {
            if (strcmp(data.time, start.c_str()) >= 0 && strcmp(data.time, end.c_str()) <= 0) {
                filteredData.push_back(data);
            }
        }

        int dataSize = filteredData.size();
        int workload = dataSize / (size - 1);
        int remainder = dataSize % (size - 1);

        int startIdx = 0;
        // Distribute data to consumer processes
        for (int i = 1; i < size; ++i) {
            int sendSize = (i <= remainder) ? (workload + 1) : workload;
            MPI_Send(&sendSize, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(filteredData.data() + startIdx, sendSize, MPI_DATA_TYPE, i, 0, MPI_COMM_WORLD);
            startIdx += sendSize;
        }

        vector<Data> sortedData(dataSize);
        startIdx = 0;
        // Collect sorted data from consumer processes
        for (int i = 1; i < size; ++i) {
            int recvSize;
            MPI_Recv(&recvSize, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(sortedData.data() + startIdx, recvSize, MPI_DATA_TYPE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            startIdx += recvSize;
        }

        // Sort the merged data based on timestamp
        sort(sortedData.begin(), sortedData.end(), sortData);

        // Aggregate the total number of cars for each traffic light
        map<int, int> congestionMap; 
        for (const auto& data : sortedData) {
            congestionMap[data.lightId] += data.carsCount;
        }

        // Print the top 3 most congested traffic lights
        cout << "Top 3 most congested traffic lights:\n";
        int count = 0;
        for (auto it = congestionMap.rbegin(); it != congestionMap.rend() && count < 3; ++it, ++count) {
            cout << "Traffic Light " << it->first << ": Number of Cars: " << it->second << " \n";
        }

        // End timing and calculate duration
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(stop - start);
        cout << "\nExecution Time Taken: " << duration.count() << " milliseconds\n" << endl;
    } else {
        int dataSize;
        MPI_Status status;
        // Receive data from producer process
        MPI_Recv(&dataSize, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        vector<Data> receivedData(dataSize);
        MPI_Recv(receivedData.data(), dataSize, MPI_DATA_TYPE, 0, 0, MPI_COMM_WORLD, &status);

        // Sort received data based on timestamp
        sort(receivedData.begin(), receivedData.end(), sortData);

        // Send sorted data back to master process
        MPI_Send(&dataSize, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(receivedData.data(), dataSize, MPI_DATA_TYPE, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Type_free(&MPI_DATA_TYPE);
    MPI_Type_free(&MPI_STRING_TYPE);
    MPI_Finalize();

    return 0;
}
