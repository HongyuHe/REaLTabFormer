from bidict import bidict
import z3


def map_to_z3var_value(var_name, value, constructor):
    var_val = None
    match constructor.label:
        case 'cidds':
            #TODO: Wrap the mapping from generated value to rule encoding in a function.
            if var_name=='Flags':
                var_val = z3.IntVal(cidds_flag_map(value))
            elif var_name=='Proto':
                var_val = z3.IntVal(cidds_proto_map(value))
            elif 'ip' in var_name.lower():
                var_val = z3.IntVal(cidds_ip_map(value))
            elif 'pt' in var_name.lower():
                value = int(value[:-2]) if value[-2:] == 'pt' else int(value)
                var_val = z3.IntVal(value)
            else:
                try:
                    if '.' in value:
                        var_val = z3.RealVal(float(value))
                    elif '_' not in value:
                        var_val = z3.IntVal(int(value))
                except ValueError:
                    var_val = value
        case 'cicids':
            domain = constructor.anuta.domains[var_name]
            if var_name == 'Protocol':
                var_val = z3.IntVal(int(value))
            elif type(domain.bounds.lb)==float or type(domain.bounds.ub)==float:
                var_val = z3.RealVal(float(value))
            else:
                var_val = z3.IntVal(int(value))
        case _:
            raise ValueError(f"Unknown dataset: {constructor.label}")
    
    assert var_val is not None, f"Rule value not mapped for {var_name}={value}"
    return var_val
        

#******************** Netflix data Domain Knowledge begins ********************
netflix_flags = ['SYN', 'ACK-SYN', 'ACK', 'ACK-PSH', 'ACK-FIN']
netflix_seqnum_increaments = [1]
netflix_tcplen_limits = [0]

def netflix_ip_map(ip: str):
    if ip in ["192.168.43.72", "75.202.11.223", 
              3232246600, 2259302873, 3232246600, 3500680008, 970557286, 1176926361]:
        #^ Client IPs
        return 0
    elif ip in ["198.38.120.153", "28.111.112.209", 
                3324410009, 3381504333, 1401727068, 3325491353, 1062720695, 3799415797]:
        #^ Server IPs
        return 1
    else:
        raise ValueError(f"Unknown IP address for Netflix dataset: {ip}")

def netflix_port_map(port: int):
    if port in [49977, 22379]:
        #^ Client ports
        return 0
    elif port in [443, 40059]:
        #^ Server ports
        return 1
    elif port >= 49152:
        #^ Application ports
        return 0
    else:
        return 1

def netflix_flags_map(flags: str):
    if flags in netflix_flags:
        return netflix_flags.index(flags)
    else:
        return -1

def netflix_proto_map(proto: str):
    if proto == 'TCP':
        return 0
    else:
        return 1

#******************** Netflix data Domain Knowledge ends ********************

popular_ports = [
    # TCP Ports
    20, 21, 22, 23, 25, 53, 80, 110, 143, 443,
    # UDP Ports
    53, 67, 68, 69, 123, 161,
    # ICMP Types
    0, 3, 5, 8, 11,
    # IGMP Types
    0x11, 0x16, 0x17, 0x22
]

#******************** CIDDS-001 Domain Knowledge begins ********************
#? Should port be a categorical variable? Sometimes we need range values (i.e., application and dynamic ports).
cidds_categorical = ['Flags', 'Proto', 'SrcIpAddr', 'DstIpAddr'] + ['SrcPt', 'DstPt']
cidds_numerical = ['Packets', 'Bytes', 'Flows', 'Duration']
cidds_ips = ['private_p2p', 'private_broadcast', 'private_any', 'public_p2p', 'dns']
cidds_ports = [0, 3, 8, 11, 22, 25, 
            #    23, #* Telnet
            #    8000, #* Seafile Server
               53, 67, 68, 80, 123, 137, 138, 443, 8080]
cidds_ports_str = [f"{port}pt" for port in cidds_ports]
cidds_ints = ['Packets', 'Bytes', 'Flows'] + cidds_categorical
cidds_reals = ['Duration']

#* Map strings to integers
cidds_ip_conversion = bidict({ip: i for i, ip in enumerate(cidds_ips)})
cidds_flags_conversion = bidict({flag: i for i, flag in enumerate(['noflags', 'flags'])})
#TODO: Change the mapping to standard NetFlow codes: 
# https://www.iana.org/assignments/protocol-numbers/protocol-numbers.xhtml
cidds_proto_conversion = bidict({proto: i for i, proto in enumerate(['TCP', 'UDP', 'ICMP', 'IGMP'])})
cidds_conversions = {
    'ip': cidds_ip_conversion,
    'flags': cidds_flags_conversion,
    'proto': cidds_proto_conversion
}
cidds_constants = {
	'ip': list(cidds_conversions['ip'].values()),
	'port': cidds_ports,
	'packet': [42, 64, 65_535], #* MTU
    'bytes': [1],
}

def cidds_proto_map(proto: str):
	return cidds_proto_conversion[proto]

def cidds_ip_map(ip: str):
    new_ip = ''
    if ip.startswith('192.168.'):
        new_ip += 'private_'
    else:        
        new_ip += 'public_'
    if '.255' in ip:
        new_ip += 'broadcast'
    else:
        new_ip += 'p2p'
    
    if ip == '0.0.0.0':
        new_ip = 'private_any'
    elif ip == '255.255.255.255':
        new_ip = 'private_broadcast'
    elif ip == 'DNS':
        new_ip = 'dns'
    
    return cidds_ip_conversion[new_ip]

def cidds_flag_map(flag: str):
	#! Don't consider the specific 'Flags' values for now
    new_flag = ''
    if flag == '......':
        new_flag = 'noflags'
    else:
        new_flag = 'flags'
    
    return cidds_flags_conversion[new_flag]
#******************** CIDDS-001 Domain Knowledge ends ********************
