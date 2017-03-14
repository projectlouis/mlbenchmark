# mlbenchmark
Benchmarking Open Source ML Frameworks (TF, Theano, MXNet, &amp; Caffe). Also Matlab Interface for Entry-Level Engineers on ML Framework

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

blend of 5 different Docker plugins were utilized to capture each containers specific CPU usage and memory
usage. Prometheus, AlertManager, Grafana, NodeExporter, and cAdvisor were compiled in separate Docker
containers and communicated via localhost proxies. These overlaying containers were segregated from one another
when collecting each of the performance metrics over each set of runs. Figure 3 shows how each container reports its
own metrics. These metrics allow for timestamps to be utilized in calculating the duration of how long each network
took to complete the training of a network
