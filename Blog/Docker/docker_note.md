
![logo](asset/logo.jpg)
# Docker for beginners

## Learn to build and deploy your distributed applications easily to the cloud with Docker

## Written by Kerry Zhou (A fourth-year CS student from Shanghai)

---

# Introduction

## What is Docker?

Wikipedia defines [Docker](https://www.docker.com) as
> an opens-source project that automates the deployment of software applications inside **containers** by providing an additinal layer of abstraction and automation of **OS-level virtualization** on Linux 

Awesome! 👍 That's a mouthful. In simple words, Docker is a tool that allows developers to easily deploy their applicatons in a sandbox(called containers) to run on the ost operating system i.e. Linux. The key benefit of Docker is that it allows users to package an application with all of its dependencies into a standardized unit for software development. 

## What are containers?
Containers offer alogical packaging mechanism in which applications can be absctrated from the environment in which they actually run. This decoupling allows container-based aplications to be deployed easily and consistently, regardlessof whether the target environment is a private data center, the public cloud, or even a developer's personal local machine. This gives developers the ability to create predictable environments that are isolated from the rest of the applications and can be run anywhere.

From an operations standpoint, apart from portability containers also give more granular control over resources giving your infrastructure improved efficiency which can result in beter utilization of your compute resources.

## What will this tutorial be teaching?

This tutorial aims to be the one-stop shop for getting your hands dirty with Docker. Apart from demystifying the Docker landscape, it'll give you hands-on experience with building and deploying your own webapps on the Cloud. We'll be using Amazon Web Services to deploy a static website, and two dynamic webapps on EC2 using Elastic Beanstalk and Elastic Container Service. Even if you have no prior experience with deployments, this tutorial should be all you need to get started. Let's get started!!!! 🏃