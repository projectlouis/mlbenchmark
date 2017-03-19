# mlbenchmark

Code snippets taken From: gh/jeffheaton [t81_558_deep_learning github], 

# Benchmarking Open Source ML Frameworks (TF, Theano, MXNet, &amp; Caffe). Also Matlab Interface for Entry-Level Engineers on ML Framework

This paper covers a brief comparative study on four frameworks—
Tensorflow, Theano, MXNet, and Caffe on several aspects: speed, utilization, and scalability
onto commercial platforms. Tensorflow achieved the overall best score and was utilized for
the language of choice for an interface between entry level engineers and an open source
framework.

The aim of this paper and likewise product, is to devise an interface that allows for
problems to be devised in Matlab and then properly ported over to Tensorflow to utilize the
extensive benefits this open-source tool provides for either GPU/CPU based hardware
configuration.

All experiments are performed on a single machine running on Windows 10 Pro (64 Bit) with Intel® Core™ i7-
4790K CPU @ 4.00GHz 3.60 GHz; Nvidia GeForce GTX 750 Ti (Ver. 378.66); 16 GiB DDR3 memory; and WD
Blue HDD. Table 1 below shows the software framework and versions utilized for the evaluation

| Framework |	Tensorflow |	Theano |	MXNet |	Caffe |
| --- | --- | --- | --- | --- |
| Version |	R1.00 |	0.8.2 |	0.9.3 |	1.0.0.rc3 |
| Core Language |	C++, Python |	Python |	C++ |	C++ |
| Interface Language |	C++, Python |	Python |	C++, Python, Julia, Matlab, Javascript, Go, R, Scala |	C, C++, Python, Matlab |
| cuDNN Support |	V5.1 |	V5 |	v.5.1 |	V5.0 |
| Python Version |	V3.5 |	V2.7 |	V2.7 |	V2.7 |
| Docker Image Pulled |	Latest:py3 |	Latest | Latest	| Latest |


Docker v.1.13.1 for Windows was chosen as the deployment
software configuration controller for each set of experiments to be
run in isolated containers via HyperV. Figure 2 shows how Docker
allows for each ML framework pre-compiled image to be invoked
on separate Docker containers on the VM and separately
monitored. Docker was allocated (of the original machine) 8 GiB
of memory and all 8 CPU Cores. No GPU runs were conducted
since Nvidia Docker currently does not have a windows interface.
Each Machine Learning Framework in Table 1 were run separately
from one another (e.g. using all CPU resources on its own and not
sharing between other frameworks).

5 different Docker plugins were utilized to capture each containers specific CPU usage and memory
usage. Prometheus, AlertManager, Grafana, NodeExporter, and cAdvisor were compiled in separate Docker
containers and communicated via localhost proxies. These overlaying containers were segregated from one another
when collecting each of the performance metrics over each set of runs. Figure 3 shows how each container reports its
own metrics. These metrics allow for timestamps to be utilized in calculating the duration of how long each network
took to complete the training of a network
