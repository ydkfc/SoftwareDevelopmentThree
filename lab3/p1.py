def selection_sort(arr):
    for i in range(len(arr)):
        min_index = i
        for j in range(i+1, len(arr)):
            if arr[j] < arr[min_index]:
                min_index = j
        arr[i], arr[min_index] = arr[min_index], arr[i]

def test():
    data = [273.5660, 1322.7970, 348.6130, 376.6244, 353.6618]
    print("Before sorting:", data)
    selection_sort(data)
    print("After sorting:", data)

test()