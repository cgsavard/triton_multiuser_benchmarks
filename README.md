# triton_multiuser_benchmarks
Benchmarking tests to be used with a triton inference server set up for a multi-user computing facility

To log into the LPC, you must open a port for jupyter and 8787 can be used for monitoring dask:
```
ssh -L localhost:8NNN:localhost:8NNN  -L localhost:8787:localhost:8787 <username>@cmslpc-sl7.fnal.gov
```

To setup:
```
git clone git@github.com:cgsavard/triton_multiuser_benchmarks.git
cd triton_multiuser_benchmarks
./setup.sh -m IMAGE
```

After setup, you can skip the previous steps and use:
```
source ./init.sh
./jupy.sh 8NNN
```

To clean the environment and start from scratch:
```
./clean.sh
```

If running dask, make sure thar you have an up-to-date grid certificate:
```
voms-proxy-init -voms cms --valid 192:00
```

