from algosdk.encoding import decode_address
from algosdk.v2client import algod
from algosdk import mnemonic, account
from algosdk import transaction
from algosdk.transaction import AssetConfigTxn
import os
from rl_model import rl_model
from pyteal import *
import smart_contract
import base64
from datetime import datetime, timedelta

# Initialize global variables
# Constants for the number of users
NUM_SELLERS = 4
NUM_BUYERS = 1
NUM_MATCHER = 1
TOTAL_SUPPLY = 2**64 -1  # Maximum supply for the asset
SCALING_FACTOR = 0.001  # scale the original value
SUCCESS = 1
FAILURE = 0

# Average block time in seconds for Algorand
AVERAGE_BLOCK_TIME = 3.5

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

# Create a new account
# Fund it with https://dispenser.testnet.aws.algodev.network/
def create_users():
    # Create a new accounts for sellers
    for i in range(NUM_SELLERS):
        # Generate a new account
        private_key, public_key = account.generate_account()
        # Generate a mnemonic from the private key
        generated_mnemonic = mnemonic.from_private_key(private_key)
        
        # Save the account information to a file
        with open("data/users.csv", "a") as f:
            f.write(f"seller_{i}, {generated_mnemonic}, {public_key}\n")
        
        print(f"Account created: {public_key}")
        print(f"Mnemonic: {generated_mnemonic}")

    # Create a new accounts for buyers 
    for i in range(NUM_BUYERS):
        # Generate a new account
        private_key, public_key = account.generate_account()
        # Generate a mnemonic from the private key
        generated_mnemonic = mnemonic.from_private_key(private_key)
        
        # Save the account information to a file
        with open("data/users.csv", "a") as f:
            private_key, public_key = account.generate_account()
            # Generate a mnemonic from the private key
            generated_mnemonic = mnemonic.from_private_key(private_key)
            f.write(f"buyer_{i}, {generated_mnemonic}, {public_key}\n")
        
        print(f"Account created: {public_key}")
        print(f"Mnemonic: {generated_mnemonic}")

    # Create a new accounts for ISO
    for i in range(NUM_MATCHER):
        private_key, public_key = account.generate_account()
        # Generate a mnemonic from the private key
        generated_mnemonic = mnemonic.from_private_key(private_key)

        # Save the account information to a file
        with open("data/users.csv", "a") as f:
            f.write(f"matcher_{i}, {generated_mnemonic}, {public_key}\n")

        print(f"Account created: {public_key}")
        print(f"Mnemonic: {generated_mnemonic}")

def get_account_info():
    seller_list = []
    buyer_list = []
    matcher = {}
    address_to_mnemonic = {}
    if (not os.path.exists("data/users.csv")):
        # Create a new accounts
        create_users()

    # Read the account information from the file
    with open("data/users.csv", "r") as f:
        lines = f.readlines()
        for line in lines:
            account_info = line.strip().split(", ")
            if len(account_info) == 3:
                
                # Store the address to mnemonic mapping
                address_to_mnemonic[account_info[2]] = account_info[1]

                if (account_info[0].startswith("seller_")):
                    seller_list.append({'Mnemonic': account_info[1], 'address': account_info[2]})
                elif (account_info[0].startswith("buyer_")):
                    buyer_list.append({'Mnemonic': account_info[1], 'address': account_info[2]})
                elif (account_info[0].startswith("matcher_")):
                    matcher = {'Mnemonic': account_info[1], 'address': account_info[2]}
            else:
                print("Invalid account information format.")
    return seller_list, buyer_list, matcher, address_to_mnemonic

# Create energy asset
def get_energy_asset(admin):
    if (not os.path.exists("data/energy_asset.csv")):
        # Create the asset
        params = algod_client.suggested_params()
        asset_name = "Energy Asset"
        unit_name = "EAS"
        total_supply = TOTAL_SUPPLY
        decimals = 0
        default_frozen = False
        manager = admin['address']
        reserve = admin['address']
        freeze = admin['address']
        clawback = admin['address']

        txn = AssetConfigTxn(
            sender=admin['address'],
            sp=params,
            total=total_supply,
            default_frozen=default_frozen,
            unit_name=unit_name,
            asset_name=asset_name,
            manager=manager,
            reserve=reserve,
            freeze=freeze,
            clawback=clawback,
            url="",
            metadata_hash=b""
        )
    
        # Sign the transaction
        private_key = mnemonic.to_private_key(admin['Mnemonic'])
        signed_txn = txn.sign(private_key)
    
        # Send the transaction
        txid = algod_client.send_transaction(signed_txn)
    
        # Wait for confirmation
        try:
            txinfo = transaction.wait_for_confirmation(algod_client, txid)
            print(f"Asset created with ID: {txinfo['asset-index']}")
        
            # Save the asset ID to a file
            with open("data/energy_asset.csv", "w") as f:
                f.write(f"{txinfo['asset-index']}\n")
        
        except Exception as e:
            print(f"Failed to create asset: {e}")

    # Get the asset id 
    asset_id = None
    with open("data/energy_asset.csv", "r") as f:
        lines = f.readlines()
        for line in lines:
            asset_id = line.strip()
            print(f"Asset ID: {asset_id}")
    return asset_id

def compile_teal(approval, clear):
    approval_teal = compileTeal(approval, mode=Mode.Application, version=6)
    clear_teal = compileTeal(clear, mode=Mode.Application, version=6)
    compiled_approval = algod_client.compile(approval_teal)
    compiled_clear = algod_client.compile(clear_teal)
    return base64.b64decode(compiled_approval['result']), base64.b64decode(compiled_clear['result'])

def get_smart_contract(admin):
    if (not os.path.exists("data/smart_contract.csv")):
        # Compile the TEAL code
        approval_program, clear_state_program = compile_teal(smart_contract.approval_program(), smart_contract.clear_state_program())
        
        # Create the application
        params = algod_client.suggested_params()
        txn = transaction.ApplicationCreateTxn(
            sender=admin['address'],
            sp=params,
            on_complete=transaction.OnComplete.NoOpOC,
            approval_program=approval_program,
            clear_program=clear_state_program,
            global_schema=transaction.StateSchema(num_uints=48, num_byte_slices=1),
            local_schema=transaction.StateSchema(num_uints=0, num_byte_slices=0)
        )
    
        # Sign the transaction
        private_key = mnemonic.to_private_key(admin['Mnemonic'])
        signed_txn = txn.sign(private_key)
    
        # Send the transaction
        txid = algod_client.send_transaction(signed_txn)
    
        # Wait for confirmation
        try:
            txinfo = transaction.wait_for_confirmation(algod_client, txid)
            print(f"Smart contract created with ID: {txinfo['application-index']}")
        
            # Save the application ID to a file
            with open("data/smart_contract.csv", "w") as f:
                f.write(f"{txinfo['application-index']}\n")
        
        except Exception as e:
            print(f"Failed to create smart contract: {e}")
    # Get the application id 
    app_id = None
    with open("data/smart_contract.csv", "r") as f:
        lines = f.readlines()
        for line in lines:
            app_id = line.strip()
            print(f"Application ID: {app_id}")
    return app_id

def has_opted_in(address, asset_id):
    account_info = algod_client.account_info(address)
    for holding in account_info.get('assets', []):
        if holding['asset-id'] == int(asset_id):
            return True
    return False

def opt_in(asset_id, sellers, buyers):
    # Opt-in sellers to the asset
    for seller in sellers:
        # Check if the seller has already opted in
        if has_opted_in(seller['address'], asset_id):
            print(f"Seller {seller['address']} has already opted in.")
            continue

        params = algod_client.suggested_params()
        txn = transaction.AssetTransferTxn(
            sender=seller['address'],
            receiver=seller['address'],  # Opt-in to the asset by sending it to themselves
            amt=0,  # Amount is zero for opt-in
            index=asset_id,
            sp=params
        )
        
        # Sign the transaction
        private_key = mnemonic.to_private_key(seller['Mnemonic'])
        signed_txn = txn.sign(private_key)
        
        # Send the transaction
        txid = algod_client.send_transaction(signed_txn)
        
        # Wait for confirmation
        try:
            txinfo = transaction.wait_for_confirmation(algod_client, txid)
            print(f"Seller {seller['address']} opted in with transaction ID: {txid}")
        
        except Exception as e:
            print(f"Failed to opt-in seller {seller['address']}: {e}")
    # Opt-in buyers to the asset
    for buyer in buyers:
        # Check if the seller has already opted in
        if has_opted_in(buyer['address'], asset_id):
            print(f"Buyer {buyer['address']} has already opted in.")
            continue
        params = algod_client.suggested_params()
        txn = transaction.AssetTransferTxn(
            sender=buyer['address'],
            receiver=buyer['address'],  # Opt-in to the asset by sending it to themselves
            amt=0,  # Amount is zero for opt-in
            index=asset_id,
            sp=params
        )
        
        # Sign the transaction
        private_key = mnemonic.to_private_key(buyer['Mnemonic'])
        signed_txn = txn.sign(private_key)
        
        # Send the transaction
        txid = algod_client.send_transaction(signed_txn)
        
        # Wait for confirmation
        try:
            txinfo = transaction.wait_for_confirmation(algod_client, txid)
            print(f"Buyer {buyer['address']} opted in with transaction ID: {txid}")
        
        except Exception as e:
            print(f"Failed to opt-in buyer {buyer['address']}: {e}")

# Get the microalgos for the transaction amount and settlement price
def get_microalgos(txn_amt, settlement_price):
    # Calculate the microalgos based on the transaction amount and settlement price
    # Assuming settlement_price is in dollars per unit of energy asset
    if settlement_price <= 0:
        return 0
    # Convert the transaction amount to microalgos
    if txn_amt <= 0:
        return 0
    # Assuming txn_amt is in units of the energy asset
    conversion_rate = 0.9 # Example conversion rate from dollars to algos
    settlement_price = settlement_price * conversion_rate  # Convert to microalg
    microalgos = int(txn_amt * settlement_price * SCALING_FACTOR)
    return microalgos

def batch_txns(txns, address_to_mnemonic, perf_stats):
    # Batch transactions to reduce the number of transactions sent to the network
    if len(txns) == 0:
        print("No transactions to batch.")
        return FAILURE
    batches = len(txns) // 16 + (1 if len(txns) % 16 > 0 else 0)
    sum_latency = 0.0
    num_txns = 0
    for i in range(batches):
        batch = txns[i*16:(i+1)*16]
        if len(batch) > 0:
            # Assign a group ID to the transactions
            gid = transaction.assign_group_id(batch)
            # Sign all transactions in the group
            for j in range(len(batch)):
                private_key = mnemonic.to_private_key(address_to_mnemonic[batch[j].sender])
                batch[j] = batch[j].sign(private_key)
            # Send the signed transaction group
            txid = algod_client.send_transactions(batch)
    
            # Wait for confirmation
            try:
                txinfo = transaction.wait_for_confirmation(algod_client, txid)
                print(f"Batch transaction ID: {txid}")
                
                # Calculate the latency and throughput
                confirmed_round = txinfo.get("confirmed-round")

                txn_body = txinfo.get("txn", {}).get("txn", {})
                first_valid = txn_body.get("fv")

                print(f"Transaction confirmed in round {confirmed_round}")
                print(f"First valid round was {first_valid}")

                fv_time = algod_client.block_info(first_valid)["block"]["ts"]
                confirmed_time = algod_client.block_info(confirmed_round)["block"]["ts"]

                latency = confirmed_time - fv_time
                num_txns = len(batch)
                perf_stats.append((latency, num_txns))
            except Exception as e:
                print(f"Failed to send batch transaction: {e}")
                return FAILURE
        
        return SUCCESS

# Main simulation
# Approach:
# 1. Get ERCOT hourly demand forecast and price forecast data as the input.
# 2. Train reinforcement learning model.
# 3. Distribute hourly demand among k consumers based imbalance least cost targets.
# 4. Get the optimal allocations from the model.
# 5. Create asset transactions to transfer energy asset from matcher to sellers.
# 6. Create application call transaction to update the completed status in the smart contract.
# 7. Trigger payments from matcher to sellers.
def main():
   sellers, buyers, matcher, address_to_mnemonic = get_account_info()
   asset_id = get_energy_asset(matcher)
   opt_in(asset_id, sellers, buyers)
   app_id = get_smart_contract(matcher)

   # Initialize the matcher, train and simulate
   model = rl_model()
   model.train()
   txn_amts, settlement_prices = model.predict(True)

   txn_date = datetime.now()
   txn_date = txn_date.replace(hour=0, minute=0, second=0, microsecond=0)

   # Create the asset transactions for each hour
   sender_address = None
   receiver_address = None
   txn_amt = None
   perf_stats = []
   for h in range(24):
       print(f"Hour {h}:")
       txn_date_str = txn_date.strftime("%Y-%m-%d %H:%M:%S")
       note_str = f"Energy transaction for hour {h} on {txn_date_str}".encode("utf-8")
       txns = []
       payment_details = []
       for i in range(NUM_SELLERS):
           # Set the sender and receiver addresses.
           if txn_amts[h][i] <= 0:
               # Battery charging is treated as supply from matcher.
               sender_address = sellers[i]['address']
               receiver_address = matcher['address']
               txn_amt = int(abs(txn_amts[h][i]))
           else:
               sender_address = matcher['address']
               receiver_address = sellers[i]['address']
               txn_amt = int(txn_amts[h][i])

           # Create a new asset transfer transaction
           params = algod_client.suggested_params()
           txn = transaction.AssetTransferTxn(
                sender=sender_address,
                receiver=receiver_address,
                amt=txn_amt,
                index=asset_id,
                note=note_str,
                sp=params
           )
           txns.append(txn)
           microalgos = get_microalgos(txn_amt, settlement_prices[h])
           payment_details.append((receiver_address, microalgos))
       txn_date = txn_date + timedelta(hours=1)

       # Create the application call transaction to record prices
       params = algod_client.suggested_params()
       app_args = [b"store_price", h.to_bytes(8, "big"), int(settlement_prices[h]).to_bytes(8, "big")]
       app_txn = transaction.ApplicationCallTxn(
                            sender=matcher['address'],
                            sp=params,
                            index=app_id,
                            on_complete=transaction.OnComplete.NoOpOC,
                            app_args=app_args)

       # Add the application call transaction
       txns.append(app_txn)
       rc1 = batch_txns(txns, address_to_mnemonic, perf_stats)

       if rc1 == SUCCESS:
           # Create the payment transactions for the matcher to sellers
           payment_txns = []
           for receiver_address, microalgos in payment_details:
               if microalgos > 0:
                    params = algod_client.suggested_params()
                    app_args = [b"release_payment", h.to_bytes(8, "big"), decode_address(receiver_address), microalgos.to_bytes(8, "big")]
                    app_txn = transaction.ApplicationCallTxn(
                            sender=matcher['address'],
                            sp=params,
                            index=app_id,
                            on_complete=transaction.OnComplete.NoOpOC,
                            app_args=app_args,
                            accounts=[receiver_address])
                    payment_txns.append(app_txn)

           rc2 = batch_txns(payment_txns, address_to_mnemonic, perf_stats)
           if rc2 ==FAILURE:
               print("Failed to release payment transactions.")
   # Print performance statistics
   if perf_stats:
         total_latency = sum(s[0] for s in perf_stats)
         total_txns = sum(s[1] for s in perf_stats)
         avg_latency = total_latency / total_txns if total_txns > 0 else 0
         throughput = total_txns / avg_latency if avg_latency > 0 else 0
         print(f"Average Latency: {avg_latency:.2f} seconds")
         print(f"Total Transactions: {total_txns}")
         print(f"Throughput: {throughput:.2f} transactions per second")
   else:
         print("No performance statistics available.")

if __name__ == "__main__":
    main()
