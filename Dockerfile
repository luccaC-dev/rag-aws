FROM public.ecr.aws/lambda/python:3.12

COPY pyproject.toml ${LAMBDA_TASK_ROOT}/
COPY rag_demo ${LAMBDA_TASK_ROOT}/rag_demo
RUN pip install --no-cache-dir "${LAMBDA_TASK_ROOT}[aws]"

CMD ["rag_demo.api.handler"]
