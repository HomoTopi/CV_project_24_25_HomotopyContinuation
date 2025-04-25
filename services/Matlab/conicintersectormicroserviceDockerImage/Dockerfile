# Dockerfile for image: conicintersector
# 
# Sample docker build command:
# docker build -f "Dockerfile" -t conicintersector "."

FROM filomba/matlabruntime:r2024b-update5

RUN apt-get update && apt-get upgrade -y

COPY ./applicationFilesForMATLABCompiler /usr/bin/mlrtapp
RUN chmod -R a+rX /usr/bin/mlrtapp/*

RUN if ["$(getent passwd appuser | cut -d: -f1)" = ""] ; then useradd -ms /bin/bash appuser ; fi
USER appuser

ENTRYPOINT ["/opt/matlabruntime/R2024b/bin/glnxa64/muserve", "-a", "/usr/bin/mlrtapp/conicIntersector.ctf"]
