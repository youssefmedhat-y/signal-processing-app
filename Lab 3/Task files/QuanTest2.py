def QuantizationTest2(file_name,Your_IntervalIndices,Your_EncodedValues,Your_QuantizedValues,Your_SampledError):
    expectedIntervalIndices=[]
    expectedEncodedValues=[]
    expectedQuantizedValues=[]
    expectedSampledError=[]
    with open(file_name, 'r') as f:
        line = f.readline()
        line = f.readline()
        line = f.readline()
        line = f.readline()
        while line:
            # process line
            L=line.strip()
            if len(L.split(' '))==4:
                L=line.split(' ')
                V1=int(L[0])
                V2=str(L[1])
                V3=float(L[2])
                V4=float(L[3])
                expectedIntervalIndices.append(V1)
                expectedEncodedValues.append(V2)
                expectedQuantizedValues.append(V3)
                expectedSampledError.append(V4)
                line = f.readline()
            else:
                break
    if(len(Your_IntervalIndices)!=len(expectedIntervalIndices)
     or len(Your_EncodedValues)!=len(expectedEncodedValues)
      or len(Your_QuantizedValues)!=len(expectedQuantizedValues)
      or len(Your_SampledError)!=len(expectedSampledError)):
        print("QuantizationTest2 Test case failed, your signal have different length from the expected one")
        return
    for i in range(len(Your_IntervalIndices)):
        if(Your_IntervalIndices[i]!=expectedIntervalIndices[i]):
            print("QuantizationTest2 Test case failed, your signal have different indicies from the expected one") 
            return
    for i in range(len(Your_EncodedValues)):
        if(Your_EncodedValues[i]!=expectedEncodedValues[i]):
            print("QuantizationTest2 Test case failed, your EncodedValues have different EncodedValues from the expected one") 
            return
        
    for i in range(len(expectedQuantizedValues)):
        if abs(Your_QuantizedValues[i] - expectedQuantizedValues[i]) < 0.01:
            continue
        else:
            print("QuantizationTest2 Test case failed, your QuantizedValues have different values from the expected one") 
            return
    for i in range(len(expectedSampledError)):
        if abs(Your_SampledError[i] - expectedSampledError[i]) < 0.01:
            continue
        else:
            print("QuantizationTest2 Test case failed, your SampledError have different values from the expected one") 
            return
    print("QuantizationTest2 Test case passed successfully")


def extract_values_from_file(file_path):
    """Extract interval indices, encoded values, quantized values, and sampled errors from a file."""
    interval_indices = []
    encoded_values = []
    quantized_values = []
    sampled_errors = []
    
    with open(file_path, 'r') as f:
        # Skip the first 3 lines (header)
        for _ in range(3):
            f.readline()
        
        # Process the data lines
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            parts = line.split(' ')
            if len(parts) == 4:
                interval_idx = int(parts[0])
                encoded = parts[1]
                quantized = float(parts[2])
                error = float(parts[3])
                
                interval_indices.append(interval_idx)
                encoded_values.append(encoded)
                quantized_values.append(quantized)
                sampled_errors.append(error)
    
    return interval_indices, encoded_values, quantized_values, sampled_errors


# Test your output against the expected output
if __name__ == "__main__":
    import os
    
    # Paths to the expected output and your output
    expected_output_path = "D:\\code\\python\\f1\\Lab 3\\Task files\\Quan2_Out.txt"
    your_output_path = "D:\\code\\python\\f1\\Lab 3\\myouts\\quan2out.txt"
    
    print(f"Loading values from your output file: {your_output_path}")
    your_interval_indices, your_encoded_values, your_quantized_values, your_sampled_errors = extract_values_from_file(your_output_path)
    
    print(f"Your interval indices: {your_interval_indices}")
    print(f"Your encoded values: {your_encoded_values}")
    print(f"Your quantized values: {your_quantized_values}")
    print(f"Your sampled errors: {your_sampled_errors}")
    
    print("\nRunning QuantizationTest2...")
    QuantizationTest2(expected_output_path, your_interval_indices, your_encoded_values, your_quantized_values, your_sampled_errors)