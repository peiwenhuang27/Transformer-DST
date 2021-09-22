import json
import argparse
import os

def getDialDomains(dialog):
    '''
    Input:
    :param dialog (dict).
    Output:
    :param domains (list).
    '''
    final_goal_list = dialog['final_goal'] # list of list of dialogue_act
    domains = []
    for da in final_goal_list:
        if da[1] not in domains:
            domains.append(da[1])
    return domains

def getDialBState(data):
    '''
    Input:
    :param data (dict).
    Output: (each item for a dialogue)
    :param usr_transcipt(list of string)
    :param sys_transcript(list of string)
    :param turn_id_all(list of int)
    :param belief_state_all(list of list of dict) 
    :param turn_label_all(list of list of list)
    :param system_acts (list).
    :param domain_all (list).
    '''

    belief_state_all = []
    turn_label_all = []
    all_messages = data["messages"]
    usr_transcript = []
    sys_transcript = []
    sys_transcript.append("")
    turn_id_all = []
    turn_id = 0
    if args.op_code == '4':
        request_slots = []
    # select_slots = []

    system_act = [[]]
    domain_all = []
    
    for i, utterance in enumerate(all_messages):

        if utterance['role'] == 'usr':
            dialogue = utterance['content'].replace('-', '~')
            usr_transcript.append(dialogue)
            turn_id_all.append(turn_id)
            turn_id += 1
            if args.op_code == '4':
                request_slots = []
                # select_slots = []
                for aPair in utterance['dialog_act']:
                    if aPair[0] == 'Request':
                        aSlot = {}
                        aSlot['slots'] = [[ aPair[1] + '-' + aPair[2], 'request']]
                        aSlot['act'] = 'request'
                        request_slots.append(aSlot)
                #   elif aPair[0] == 'Select':
                #     aSlot = {}
                #     aSlot['slots'] = [[ aPair[1] + '-' + aPair[2], aPair[3]]]
                #     aSlot['act'] = 'select'
                #     select_slots.append(aSlot)
        else:
            belief_state = []
            turn_label = []
            set_domain = set()

            # remove the last utterance
            if i != (len(all_messages) - 1):
                dialogue = utterance['content'].replace('-', '~')
                sys_transcript.append(dialogue)
                system_act.append([])
                for dialog_act in utterance['dialog_act']:
                    # if i == 1:
                    #   set_domain.add(dialog_act[1])
                    if dialog_act[2] == 'none':
                        continue
                    if dialog_act[3] == "":
                        system_act[-1].append(dialog_act[2])
                    else:
                        system_act[-1].append([dialog_act[2], dialog_act[3]])

            for domain in utterance['sys_state_init']:
                for slot in utterance['sys_state_init'][domain].keys():
                    aSlot = {}
                    # get slots
                    if (slot != 'selectedResults') & (utterance['sys_state_init'][domain][slot] != ""):
                        # if slot != '酒店设施':
                        aSlot['slots'] = [[ domain + '-' + slot, utterance['sys_state_init'][domain][slot].replace('-', '~')]]
                        # else:
                        #   for item in utterance['sys_state_init'][domain][slot].split(' '):
                        #     aSlot['slots'] = [[domain + '-' + item.replace('-', '~'), 'yes']]
                        aSlot['act'] = 'inform'
                        belief_state.append(aSlot)

            if args.op_code == "4":
                belief_state = belief_state + request_slots
                # belief_state = belief_state + select_slots

            # diff between this and last turn (只看增加不看減少)
            if belief_state_all != []:
                for this_turn in belief_state:
                    if this_turn not in belief_state_all[-1]:
                        turn_label.append(this_turn['slots'][0])
                        set_domain.add(this_turn['slots'][0][0].split("-")[0])
                if len(set_domain) == 0:
                    set_domain = domain_all[-1]
            else:
                for aPair in belief_state:
                    turn_label.append(aPair['slots'][0])
                    set_domain.add(aPair['slots'][0][0].split("-")[0])

            domain_all.append([*set_domain, ])
            belief_state_all.append(belief_state)
            turn_label_all.append(turn_label)

    return (usr_transcript, sys_transcript, turn_id_all, belief_state_all, turn_label_all, system_act, domain_all)

def create_dials(data):
    dials = []

    for i, dialIdx in enumerate(data):
        domains = getDialDomains(data[dialIdx])
        usr_transcript, sys_transcript, turn_id_all, belief_state_all, turn_label_all, system_act, domain = getDialBState(data[dialIdx])

        # print(len(usr_transcript), len(belief_state_all), len(system_act), len(turn_label_all), len(sys_transcript), len(turn_id_all))
        output = dict()
        dial_list = []
        for i in range(len(usr_transcript)):
            dialog = dict()
            dialog['system_transcript'] = sys_transcript[i]
            dialog['turn_idx'] = turn_id_all[i]
            dialog['belief_state'] = belief_state_all[i]
            dialog['turn_label'] = turn_label_all[i]
            dialog['transcript'] = usr_transcript[i]
            dialog['system_acts'] = system_act[i]
            dialog['domain'] = domain[i]
            dial_list.append(dialog)
            # dialIdx (str) is the dialogue_idx itself, no need for further processing
        output['dialogue_idx'] = dialIdx
        output['domains'] = domains
        output['dialogue'] = dial_list
        
        dials.append(output)
    return dials

def main(args):

    train_file = open(args.train_rawData_path)
    data_train = json.load(train_file) # dict object
    test_file = open(args.test_rawData_path)
    data_test = json.load(test_file) 
    dev_file = open(args.dev_rawData_path)
    data_dev = json.load(dev_file)

    test_dials = create_dials(data_test)
    train_dials = create_dials(data_train)
    dev_dials = create_dials(data_dev)

    
    if args.op_code == "4":
        for name in ['train_dials', 'test_dials', 'dev_dials']:
            with open(os.path.join(args.save_dir, f'{name}_request.json'), 'w', encoding='utf-8') as fp:
                exec(f'json.dump({name}, fp, ensure_ascii=False, indent=4)')
            print(os.path.join(args.save_dir, f'{name}_request.json'))
        
    else:
        for name in ['train_dials', 'test_dials', 'dev_dials']:
            with open(os.path.join(args.save_dir, f'{name}.json'), 'w', encoding='utf-8') as fp:
                exec(f'json.dump({name}, fp, ensure_ascii=False, indent=4)')
            print(os.path.join(args.save_dir, f'{name}.json'))




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_rawData_path", default="", type=str)
    parser.add_argument("--test_rawData_path", default="", type=str)
    parser.add_argument("--dev_rawData_path", default="", type=str)

    parser.add_argument("--save_dir", default="./cleanData", type=str)

    parser.add_argument("--op_code", default="3-2", type=str)

    args = parser.parse_args()
    main(args)
