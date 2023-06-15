import random


def _perturb(old_part_list, change_idx, tochange_part):
    new_part_list = old_part_list.copy()
    new_part_list[change_idx] = tochange_part

    return new_part_list

def context_text_optimization(start_text, start_part_list, idx_part_list, cand_dict, eval_f, change_rate=1, max_iter=1, do_random=False, random_factor=0.5):
    # algorithm 2: greedy opt
    cur_query_num = 0
    cur_edit_num = 0

    # orig_word_list = word_list.copy()
    if " ".join(start_part_list) != start_text:
        print('Text:', start_text)
        print('Join Text:', " ".join(start_part_list))

    # old_text = start_text

    best_text = start_text
    best_score = eval_f([start_text])[0]
    best_part_list = start_part_list.copy()

    # print('Init best score', best_score)

    if change_rate > 1:
        assert isinstance(change_rate, int)
        op_num = change_rate
    else:
        op_num = max(1, round(len(best_part_list) * change_rate))
    # diff_indices = []

    for _ in range(max_iter):
        new_text_list = []
        operation_list = []

        # generate text list (one-step op)
        for part_idx, orig_part in idx_part_list:
            k_ = (part_idx, orig_part)
            cur_cand_list = cand_dict[k_]

            assert len(cur_cand_list) > 0

            # swap
            for cand_ in cur_cand_list:
                if cand_ == best_part_list[part_idx]:
                    continue
                new_part_list = _perturb(best_part_list, part_idx, cand_)
                new_text_list.append(" ".join(new_part_list))
                operation_list.append((part_idx, cand_))

            # swap back
            if best_part_list[part_idx] != orig_part:
                new_part_list = _perturb(best_part_list, part_idx, orig_part)
                new_text_list.append(" ".join(new_part_list))
                operation_list.append((part_idx, orig_part))

        # evaluate
        new_text_score = eval_f(new_text_list)
        cur_query_num += len(operation_list)

        # new_text_delta_score = [s - best_score for s in new_text_score] # find difference

        # stochastic ordering
        if do_random:
            random_range = max(new_text_score) - best_score
            new_text_delta_score = [s - best_score + (random.random() - 0.5) * random_range * random_factor for s
                                    in
                                    new_text_score]
        else:
            new_text_delta_score = [s - best_score for s in new_text_score]

        op_score = zip(operation_list, new_text_delta_score)

        # sort based on one-step operation
        sorted_op_score = sorted(op_score, key=lambda x: x[1], reverse=True)

        # do the operation based on score, large -> small
        cur_change_indices = set()
        part_list_after_op = best_part_list.copy()
        for op_, score_ in sorted_op_score:
            part_idx_, replace_part_ = op_
            if part_idx_ in cur_change_indices:  # skip position that already done operations
                continue

            # print(f'Do OP at {w_idx_}: {sub_sent_list_after_op[w_idx_]} -> {w_}')
            part_list_after_op[part_idx_] = replace_part_

            cur_change_indices.add(part_idx_)

            if len(cur_change_indices) >= op_num:
                break

            if score_ < 0:
                break

        cur_edit_num += len(cur_change_indices)

        # update best text
        text_after_op = " ".join(part_list_after_op)
        # best_text = text_after_op
        score_after_op = eval_f([text_after_op])[0]
        if score_after_op > best_score:
            best_score = score_after_op
            best_part_list = part_list_after_op
            best_text = text_after_op
            # print('Current best text:', best_text, f'(Best Score: {best_score:.2f}')
        else:
            break

        if len(cur_change_indices) == 0:
            # print('This iter does not change.')
            break

    # if best_text == old_text:
    #     print('This round does not change Text')

    return best_text, best_score, cur_query_num, cur_edit_num
