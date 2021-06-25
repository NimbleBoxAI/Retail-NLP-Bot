# This function contains modules that can be used to extract relevant parts of an unstructured query.

def get_name(gpt, query):
    
    prompt = f"""Input: "Raghav Subbarayan, 2107, Panchratna, Opera House, Mumbai, Maharashtra-400004 Phone Number: 02223614567"

Name --> Raghav Subbarayan

###

Input: "Sudhamay Lavanya, 20 W 34th St, New York, NY, USA"

Name --> Sudhamay Lavanya

###

Input: "Dheerendra Narasimha, 22, Beech Glen Street, Boston, MA 02115, USA, Phone:+17619781234 "

Name --> Dheerendra Narasimha

###

Input: "Anuja E302 Casa Grande The Address, Easwaran Street Karapakkam, OMR Chennai"

Name --> Anuja

###

Input: "Shantala 1 multi bead neckware"

Name --> Shantala

###

Input: "{query}"

Name -->"""
    out = gpt(prompt, n=500, temp=0.5, top_k=None, stop_sequence='###', r=1)
    return out.decoded[0][len(prompt):].split('\n')[0].strip()


def get_city(gpt, query):
    
    prompt = f"""Here we are trying to extract details of an address, such as name, house number, street name, city, state, etc

Input: "2107, Panchratna, Opera House, Mumbai, Maharashtra-400004 Phone Number: 02223614567"

City --> Mumbai, Maharashtra

###

Input: "20 W 34th St, New York, NY 10001, USA"

City --> New York, NY

###

Input: "22, Beech Glen Street, Boston, MA 02115, USA, Phone:+17619781234 "

City --> Boston, MA

###

Input: "{query}"

City -->"""
    out = gpt(prompt, n=500, temp=0.5, top_k=None, stop_sequence='###', r=1)
    return out.decoded[0][len(prompt):].split('\n')[0].strip()


def get_pincode(gpt, query):
    
    prompt = f"""Input: "2107, Panchratna, Opera House, Mumbai, Maharashtra-400004 Phone Number: 02223614567"

Pin Code --> 400004

###

Input: "20 W 34th St, New York, NY, USA"

Pin Code --> ?

###

Input: "22, Beech Glen Street, Boston, MA 02115, USA, Phone:+17619781234 "

Pin Code --> 02115

###

Input: "Anuja Jacob E302 Casa Grande The Address, Easwaran Street Karapakkam, OMR Chennai"

Pin Code --> ?

###

Input: "{query}"

Pin Code --> """
    out = gpt(prompt, n=500, temp=0.5, top_k=None, stop_sequence='###', r=1)
    return out.decoded[0][len(prompt):].split('\n')[0].strip()


def get_phone(gpt, query):
    
    prompt = f"""Input: "Raghav Subbarayan, 2107, Panchratna, Opera House, Mumbai, Maharashtra-400004 Phone Number: 02223614567"

Phone Number: 02223614567

###

Input: "Sudhamay Lavanya, 20 W 34th St, New York, NY, USA"

Phone Number: ?

###

Input: "Dheerendra Narasimha, 22, Beech Glen Street, Boston, MA 02115, USA, Phone:+17619781234 "

Phone: +17619781234

###

Input: "Anuja Jacob E302 Casa Grande The Address, Easwaran Street Karapakkam, OMR Chennai"

Phone Number: ?

###

Input: "{query}"

Phone Number:"""
    out = gpt(prompt, n=500, temp=0.5, top_k=None, stop_sequence='###', r=1)
    return out.decoded[0][len(prompt)-1:].split('\n')[0].strip()


def get_orders(gpt, query):
    
    prompt = f"""Input: "Order Janaki Houston Parkar Polka 1-2 years - 4pcs Parkar Polka 6 year - 1 pc"

Order ->
1) Parkar Polka 1-2 years - 4pcs
2) Parkar Polka 6 year - 1 pc

###

Input: "Order Tara Vishwanath Houston 1) White skirt size 2-4 hasiru 2) Parkar Polka size 1-2 Maroon 3)cushion cover barcode 16"*16".-2pcs."

Order ->
1) White skirt size 2-4 hasiru
2) Parkar Polka size 1-2 Maroon
3) cushion cover barcode 16"*16".-2pcs

###

Input: "Uma Deole - 2 red shirts"

Order ->
1) 2 red shirts

###

Input: "{query}"

Order ->
1)"""
    out = gpt(prompt, n=500, temp=0.8, top_k=None, stop_sequence='###', r=1)
    return out.decoded[0][len(prompt)-2:-5]