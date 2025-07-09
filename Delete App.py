from algosdk.v2client import algod
from algosdk import account, mnemonic, transaction

# Algod client setup (TestNet)
algod_address = "https://testnet-api.algonode.cloud"
algod_token = ""  # Often empty for public nodes
algod_client = algod.AlgodClient(algod_token, algod_address)

# Creator's account
creator_mnemonic = "bonus person knife side screen price tourist winter arrive sock bless uncle fence primary vocal devote song color fiscal enlist can keep early about step"
creator_private_key = mnemonic.to_private_key(creator_mnemonic)
creator_address = "BQ5LGQGJBX7DKQCHGAHTBZWCBZCEIS6YEOWMUXM7W2KNYTLCEB7SIH3VEQ"

# Application ID to destroy
app_id = 742558460  # Replace with your app ID

from algosdk.logic import get_application_address
print(get_application_address(app_id))  # replace with your app ID

# Get suggested parameters
params = algod_client.suggested_params()

# Create DeleteApplication transaction
txn = transaction.ApplicationDeleteTxn(
    sender=creator_address,
    sp=params,
    index=app_id
)

# Sign and send the transaction
signed_txn = txn.sign(creator_private_key)
txid = algod_client.send_transaction(signed_txn)
print("Sent DeleteApplication transaction with txID:", txid)

# Wait for confirmation
from algosdk.transaction import wait_for_confirmation
wait_for_confirmation(algod_client, txid, 4)
print(f"Application {app_id} has been successfully deleted.")

