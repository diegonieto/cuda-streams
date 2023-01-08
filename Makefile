all
	nvcc streams.cu -o streams
	nvcc no_streams.cu -o no_streams

clean:
	rm no_streams streams