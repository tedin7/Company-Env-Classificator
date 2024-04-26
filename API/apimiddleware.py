from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from database_setup import SessionLocal, APILog
import json
import logging

class APILoggerMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        # Skip logging for the /api_logs/ endpoint or any other specific endpoints
        if "/api_logs/" in request.url.path or "/other_endpoint/" in request.url.path:
            return await call_next(request)

        # Read and store request body for logging and re-inject if necessary
        request_body = await request.body()
        request.state.body = request_body  # Store the body in the request state
        
        # Create a custom receive that returns the stored body
        async def custom_receive():
            yield {'type': 'http.request', 'body': request_body}

        # Replace the request's receive method with the custom receive
        request._receive = custom_receive

        # Process the request and capture the response
        response = await call_next(request)

        # Access the response body properly
        response_body = b''.join([chunk async for chunk in response.__dict__['body_iterator']])
        response_content = response_body.decode(response.charset)

        # Only log if the response status is an error or other specific conditions
        if response.status_code >= 400:
            db = SessionLocal()  # Create a new session for each request
            try:
                self.log_api_call(
                    db,
                    path=request.url.path,
                    method=request.method,
                    request_data=request_body.decode(),
                    status_code=response.status_code,
                    response_data=response_content
                )
            finally:
                db.close()

        # Return the original response, re-constructed from the body iterator
        return Response(content=response_body, status_code=response.status_code, headers=dict(response.headers))

    def log_api_call(self, db, path, method, request_data, status_code, response_data):
        try:
            log_entry = APILog(
                request_data=json.dumps({
                    "path": path,
                    "method": method,
                    "request_data": request_data
                }),
                response_data=json.dumps({
                    "status_code": status_code,
                    "response_data": response_data
                })
            )
            db.add(log_entry)
            db.commit()
        except Exception as e:
            logging.error(f"Failed to log API call: {e}")
