
# Python3 + openCV 3 Image
FROM jjanzic/docker-python3-opencv:contrib-opencv-3.4.1

# Dependencies
RUN pip install matplotlib

# Setup Matplotlib
COPY ./devops/matplotlibrc /root/.config/matplotlib/matplotlibrc

# Environment
ENV APP_ROOT="/var/www"

# Set the application root
WORKDIR ${APP_ROOT}

# Listen to port
EXPOSE 8080

# Setup application
COPY ./devops/start.sh /opt/start.sh
RUN chmod 755 /opt/start.sh

# Start application
ENTRYPOINT [ "/opt/start.sh" ]