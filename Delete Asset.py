from algosdk.v2client import algod
from algosdk import mnemonic, account
from algosdk.v2client import indexer
from algosdk import transaction
from algosdk.transaction import AssetDestroyTxn

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

# Creator's credentials
creator_mnemonic = "bonus person knife side screen price tourist winter arrive sock bless uncle fence primary vocal devote song color fiscal enlist can keep early about step"
creator_private_key = mnemonic.to_private_key(creator_mnemonic)
creator_address = "BQ5LGQGJBX7DKQCHGAHTBZWCBZCEIS6YEOWMUXM7W2KNYTLCEB7SIH3VEQ"

# Asset ID to destroy
asset_id = 740043900  # Replace with your ASA ID

indexer_client = indexer.IndexerClient("", "https://testnet-idx.algonode.cloud")

# Get all accounts holding this asset
response = indexer_client.accounts(asset_id=asset_id)
accounts = response['accounts']
for acct in accounts:
    print(f"{acct['address']} holds {acct['assets'][0]['amount']} units")
    # Claw back if necessary
    if acct['assets'][0]['amount'] > 0 and acct['address'] != creator_address:
        print(f"Clawing back {acct['assets'][0]['amount']} units from {acct['address']}")
        from_address = acct['address']
        amount = acct['assets'][0]['amount']
        params = algod_client.suggested_params()
        txn = transaction.AssetTransferTxn(
            sender=creator_address,
            sp=params,
            receiver=creator_address,
            amt=amount,
            index=asset_id,
            revocation_target=from_address  # this is the account you're clawing back from
        )

        signed_txn = txn.sign(creator_private_key)
        txid = algod_client.send_transaction(signed_txn)
        transaction.wait_for_confirmation(algod_client, txid, 4)
        print(f"Clawed back {amount} from {from_address}")


#Get suggested params
params = algod_client.suggested_params()

# Create asset destroy transaction
txn = AssetDestroyTxn(
    sender=creator_address,
    sp=params,
    index=asset_id
)

# Sign the transaction
signed_txn = txn.sign(creator_private_key)

# Send the transaction
txid = algod_client.send_transaction(signed_txn)
print("Transaction ID:", txid)

# Wait for confirmation
from algosdk.transaction import wait_for_confirmation
wait_for_confirmation(algod_client, txid, 4)
print(f"Asset {asset_id} has been successfully destroyed.")
