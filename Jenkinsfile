pipeline {
    agent any
    environment {
        DOCKER_IMAGE = "afnannaseem837/mlops-a01"
        DOCKERHUB_CREDENTIALS = credentials('dockerhub-creds') // From Jenkins credentials
    }
    stages {
        stage('Checkout Code') {
            steps {
                git branch: 'main', url: 'https://github.com/amnashahid31/mlops_assignment_1.git'
            }
        }

        stage('Build Docker Image') {
            steps {
                script {
                    sh 'sudo docker build -t fatimaqurban/mlops-assignment-1:v1 .'
                }
            }
        }
        stage('Push to Docker Hub') {
            steps {
                script {
                    withDockerRegistry([url: "https://registry.hub.docker.com", credentialsId: "dockerhub-creds"]) {
                        sh 'docker login -u fatimaqurban --password-stdin < credentialsId'
                        sh 'docker push fatimaqurban/mlops-assignment-1:v1'
                    }
                }
            }
        }

    }
    post {
        success {
            emailext (
                subject: 'Deployment Successful',
                body: """
                    The Docker image has been built and pushed to Docker Hub.
                    Image: ${DOCKER_IMAGE}:${env.BUILD_ID}
                """,
                to: 'aghaaleedurrani54@gmail.com' // Replace with admin email
            )
        }
    }
}