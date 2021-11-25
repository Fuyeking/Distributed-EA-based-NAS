import random
from operator import attrgetter
from deap import tools

# 定义自己的统计模型规模的函数
import torch
import torch.nn as nn

from collections import OrderedDict
import numpy as np


def summary(model, input_size, batch_size=-1, device="cuda"):
    def register_hook(module):

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
                not isinstance(module, nn.Sequential)
                and not isinstance(module, nn.ModuleList)
                and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
    # print(type(x[0]))

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()
    ''' 打印
    print("----------------------------------------------------------------")
    line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
    print(line_new)
    print("================================================================")
    '''
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]
        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        # print(line_new)打印

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size
    ''' 打印
    print("================================================================")
    print("Total params: {0:,}".format(total_params))
    print("Trainable params: {0:,}".format(trainable_params))
    print("Non-trainable params: {0:,}".format(total_params - trainable_params))
    print("----------------------------------------------------------------")
    print("Input size (MB): %0.2f" % total_input_size)
    print("Forward/backward pass size (MB): %0.2f" % total_output_size)
    print("Params size (MB): %0.2f" % total_params_size)
    print("Estimated Total Size (MB): %0.2f" % total_size)
    print("----------------------------------------------------------------")
    '''
    # return summary
    return total_params_size


# 定义新的选择函数
def select_best(individuals, k, fit_attr1="fitness", fit_attr2="model_size"):
    pop = sorted(individuals, key=attrgetter(fit_attr1), reverse=True)
    return sorted(pop, key=attrgetter(fit_attr2), reverse=True)[:k]


# 把一个list划分为多个list
def split_gen(genlist, n):
    for i in range(0, len(genlist), n):
        yield genlist[i:i + n]


# 自定义突变函数
def random_mutation_fun(individual, func, gen_l, indpb=0.4):
    for i in range(len(individual)):
        if random.random() < indpb:
            individual[i] = func[i % gen_l]()
    return individual,


# 自定义杂交函数
def random_mate_fun(ind1, ind2, gen_size, num):
    ind1_start = random.randint(0, num - 1)
    ind1_start = ind1_start * gen_size
    ind2_start = random.randint(0, num - 1)
    ind2_start = ind2_start * gen_size
    ind1[ind1_start:ind1_start + gen_size], ind2[ind2_start:ind2_start + gen_size] \
        = ind2[ind2_start:ind2_start + gen_size], ind1[ind1_start:ind1_start + gen_size]
    return ind1, ind2


def varAnd(population, toolbox, cxpb, mutpb, gen_size, block_size, func):
    offspring = [toolbox.clone(ind) for ind in population]

    # Apply crossover and mutation on the offspring
    for i in range(1, len(offspring), 2):  # crossover
        if random.random() < cxpb:
            offspring[i - 1], offspring[i] = random_mate_fun(offspring[i - 1],
                                                             offspring[i], gen_size, block_size)
            del offspring[i - 1].fitness.values, offspring[i].fitness.values

    for i in range(len(offspring)):  # mutation
        if random.random() < mutpb:
            offspring[i], = random_mutation_fun(offspring[i], func, gen_size)
            del offspring[i].fitness.values

    return offspring


# 自定义算法
def ea_simple(population, toolbox, cxpb, mutpb, ngen, gen_size, block_size, func, growth_steps, verbose=__debug__):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals']
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    file_name = "gen" + str(0) + ".txt"
    fp = open(file_name, "w")
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
        model_size = toolbox.calc_model_size(ind, growth_steps, gen_size)
        fp.write('Model:%s,Best accuracy:%0.3f%%,model size:%0.3f M' % (str(ind), fit[0], model_size))
        fp.write('\n')
        fp.flush()
    fp.closed
    logbook.record(gen=0, nevals=len(invalid_ind))
    if verbose:
        print(logbook.stream)
    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population) // 2)
        # Vary the pool of individuals
        offspring = varAnd(offspring, toolbox, cxpb, mutpb, gen_size, block_size, func)
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        file_name = "gen" + str(gen) + ".txt"
        fp = open(file_name, "w")
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            model_size = toolbox.calc_model_size(ind, growth_steps, gen_size)
            fp.write('Model:%s,accuracy:%.3f%%,model size:%f M' % (str(ind), fit[0], model_size))
            fp.write('\n')
            fp.flush()
        fp.closed
        # Replace the current population by the offspring
        population[:] = offspring
        # Append the current generation statistics to the logbook
        logbook.record(gen=gen, nevals=len(invalid_ind))
        if verbose:
            print(logbook.stream)

    return population, logbook


# sample evolution
def dis_ea_simple(population, toolbox):
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    return population
