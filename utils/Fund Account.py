from algosdk.v2client import algod
from algosdk.transaction import PaymentTxn
from algosdk.logic import get_application_address
from algosdk import mnemonic, account

# Replace these values with your node's address
# free service does not require tokens
algod_address = "https://testnet-api.4160.nodely.dev"
algod_indexer = "https://testnet-idx.4160.nodely.dev/"
algod_token = ""

# Initialize the algod client
algod_client = algod.AlgodClient(algod_token, algod_address)

# Get the node status
try:
    status = algod_client.status()
    print("Node status:", status)
except Exception as e:
    print(f"Failed to get node status: {e}")

creator_mnemonic =  "slow month cement myth nest cricket surround open clump pulse palm wide nature laugh three foster kangaroo seven jar mutual head blind require able quick"
creator_private_key = mnemonic.to_private_key(creator_mnemonic)
creator_address = "ZX47ZW2TIEBML72LM6ES7PPH3ZVOPX6227UQZAQIXHURAK6GEDPMZCTOOY"

params = algod_client.suggested_params()
app_addr = get_application_address(742561048)

txn = PaymentTxn(
    sender=creator_address,
    sp=params,
    receiver=app_addr,
    amt=100000  # e.g., fund
)

# Sign and send the transaction
signed_txn = txn.sign(creator_private_key)
txid = algod_client.send_transaction(signed_txn)
print("Sent fund transaction with txID:", txid)

# Wait for confirmation
from algosdk.transaction import wait_for_confirmation
wait_for_confirmation(algod_client, txid, 4)
print(f"Account {app_addr} has been funded.")
