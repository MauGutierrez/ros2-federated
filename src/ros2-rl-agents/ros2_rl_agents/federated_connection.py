import json
import requests

class FederatedConnection():
    def __init__(self):
        pass
    
    def add_agent_to_network(self):
        payload = {
            "id": "agent1",
            "public_key": "test",
            "session_id": "agent1-123",
            "address": "testing"
        }
        response = None
        try:
            response = requests.post(
                "http://127.0.0.1:8080/api/v1/register",
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
            )

        except Exception as e:
            print("Failed to register agent to network")
        
        return response

    def add_local_update(self, message):
        response = None
        try:
            response = requests.post(
                "http://127.0.0.1:8080/api/v1/add-local",
                headers={"Content-Type": "application/json"},
                data=json.dumps(message)
            )
        except Exception as e:
            print("Failed to add local update")
        
        return response
    
    def get_global(self):
        response = None
        try:
            response = requests.get(
                "http://127.0.0.1:8080/api/v1/get-global",
            )
        except Exception as e:
            print("Failed to get Global from Server")
        
        return response