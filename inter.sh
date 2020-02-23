#!/bin/bash
salloc --account=def-cogdrive --gres=gpu:2 --cpus-per-task=20 --mem=47G --time=0-02:00:00

#script for running a job interactively
#useful for testing before running a long job

