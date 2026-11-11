FROM pytorch/pytorch:2.11.0-cuda13.0-cudnn9-devel@sha256:6e8a7a6dedf900096f90190f66f988e7e658cda4f1e6cbc7c17e3a38980a4f89
WORKDIR /pilot
COPY requirements.txt /tmp/pilot-requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/pilot-requirements.txt
ENTRYPOINT ["python", "-u", "/pilot/run_pilot.py"]
CMD ["--execute"]
