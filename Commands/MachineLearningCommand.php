<?php
/*
 *  Copyright 2025.  Baks.dev <admin@baks.dev>
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 *  of this software and associated documentation files (the "Software"), to deal
 *  in the Software without restriction, including without limitation the rights
 *  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *  copies of the Software, and to permit persons to whom the Software is furnished
 *  to do so, subject to the following conditions:
 *
 *  The above copyright notice and this permission notice shall be included in all
 *  copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NON INFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 *  THE SOFTWARE.
 */

declare(strict_types=1);

namespace BaksDev\Machine\Learning\Commands;


use BaksDev\Machine\Learning\Matrix\Brainy;
use Symfony\Component\Console\Attribute\AsCommand;
use Symfony\Component\Console\Command\Command;
use Symfony\Component\Console\Input\InputArgument;
use Symfony\Component\Console\Input\InputInterface;
use Symfony\Component\Console\Output\OutputInterface;
use Symfony\Component\Console\Style\SymfonyStyle;

#[AsCommand(
    name: 'baks:machine:learning',
    description: 'Описание комманды'
)]
class MachineLearningCommand extends Command
{
    public function __construct(private readonly Brainy $BrainyService)
    {
        parent::__construct();
    }

    protected function configure(): void
    {
        $this->addArgument('argument', InputArgument::OPTIONAL, 'Описание аргумента');
    }

    protected function execute(InputInterface $input, OutputInterface $output): int
    {
        $io = new SymfonyStyle($input, $output);

        dd(46546546); /* TODO: удалить !!! */


        // tanh     : 30000   0.01    3   -1
        // sigmoid  : 30000   0.01    3   -1
        // relu     : 3000    0.01    3   0

        // choose the tot number of epochs
        $epochs = 30000;
        // choose the learning rate
        $learning_rate = 0.001;
        // numbers of hidden neurons of the first (and only one) layer
        $hidden_layer_neurons = 3;
        // activation functions: relu , tanh , sigmoid
        $activation_fun = 'relu';

        $brain = $this->BrainyService
            ->setLearningRate($learning_rate)
            ->setActivationFun($activation_fun);

        // Это матрица ввода XOR
        // Не забудьте заменить нули на -1, когда вы используете Tanh или Sigmoid
        $xor_in = [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ];

        // Это выход XOR
        // Не забудьте заменить нули на -1, когда вы используете Tanh или Sigmoid
        $xor_out = [
            [1],
            [0],
            [0],
            [1],
        ];


        $input_neurons = count($xor_in[0]);
        $output_neurons = count($xor_out[0]);

        // getting the W1 weights random matrix (layer between input and the hidden layer) with size 2 x $hidden_layer_neurons
        $w1 = $w1_before = $brain->getRandMatrix($input_neurons, $hidden_layer_neurons);

        // getting the W2 weights random vector (layer between hidden layer and output) with size $hidden_layer_neurons x 1
        $w2 = $w2_before = $brain->getRandMatrix($hidden_layer_neurons, $output_neurons);

        // getting the B1 bies random vector with size $hidden_layer_neurons
        $b1 = $b1_before = $brain->getRandMatrix($hidden_layer_neurons, 1);

        // getting the B2 bies random vector. The size is 1x1 because there is only one output neuron
        $b2 = $b2_before = $brain->getRandMatrix($output_neurons, 1);


        $w1 = $w1_before = [
            [-0.43, -0.21, -0.58],
            [-0.05, 0.84, -0.07],
        ];

        $b1 = $b1_before = [
            [-0.86],
            [-0.76],
            [0.93],
        ];

        $w2 = $w2_before = [
            [0.61],
            [0.02],
            [0.94],
        ];
        $b2 = $b1_before = [
            [-0.88],
        ];


        // this is for the chart
        $graph = [];
        $denom = 0;
        $correct = 0;
        $points_checker = $epochs / 100 * 4;
        if($points_checker < 10)
            $points_checker = 10;


        // preparing the arrays
        foreach($xor_in as $index => $input)
        {
            $xor_in[$index] = $brain->arrayTranspose($input);
            $xor_out[$index] = $brain->arrayTranspose($xor_out[$index]);
        }


        $execution_start_time = microtime(true);

        for($i = 0; $i < $epochs; $i++)
        {
            foreach($xor_in as $index => $input)
            {
                // forward the input and get the output
                $forward_response = $brain->forward($input, $w1, $b1, $w2, $b2);

                // backprotagating the error and finding the new weights and biases
                $new_setts = $brain->backPropagation($forward_response, $input, $xor_out[$index], $w1, $w2, $b1, $b2);
                $w1 = $new_setts['w1'];
                $w2 = $new_setts['w2'];
                $b1 = $new_setts['b1'];
                $b2 = $new_setts['b2'];

                // this is only for che accuracy chart
                $f1 = round($brain->getScalarValue($forward_response['A']), 2);
                $f2 = round($brain->getScalarValue($xor_out[$index]), 2);
                if($f2 < 0)
                    $f2 = 0;
                if($f1 == $f2)
                    $correct++;
                $denom++;

            } // end foreach

            // this is only for che accuracy chart
            if(!($i % $points_checker))
            {
                $graph[] = $rate = $correct / $denom;
                $denom = 0;
                $correct = 0;
            }

        } // end for $epochs


        $execution_time = round(microtime(true) - $execution_start_time, 2);


        $g_labes = $g_vals = '';
        foreach($graph as $num => $val)
        {
            $g_labes .= ($num * $points_checker).',';
            $g_vals .= (round($val, 2)).',';
        }
        $g_labes = trim($g_labes, ',');
        $g_vals = trim($g_vals, ',');


        $io->success('baks:machine:learning');

        return Command::SUCCESS;
    }
}
