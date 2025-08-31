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
use Phpml\Classification\SVC;
use Phpml\FeatureExtraction\TfIdfTransformer;
use Phpml\FeatureExtraction\TokenCountVectorizer;
use Phpml\ModelManager;
use Phpml\SupportVectorMachine\Kernel;
use Phpml\Tokenization\WordTokenizer;
use SebastianBergmann\CodeCoverage\Report\PHP;
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

    protected function ___execute(InputInterface $input, OutputInterface $output): int
    {
        $io = new SymfonyStyle($input, $output);


        // Параметры нейронной сети
        $epochs = 30000;
        $learning_rate = 0.05;
        $hidden_layer_neurons = 12;
        $activation_fun = 'sigmoid'; //  ['relu', 'sigmoid', 'tanh'])


        $brain = new Brainy()
            ->setLearningRate($learning_rate)
            ->setActivationFun($activation_fun);


        // Новые категории
        $product_categories = [
            'ноутбук',
            'телефон',
            'планшет',
        ];

        $request_types = [
            'стоимость',
            'наличие',
            'общее',
        ];

        $all_categories = [];

        // Создаем комбинированные категории
        foreach($product_categories as $product)
        {
            foreach($request_types as $type)
            {
                $all_categories[] = $product.'_'.$type;
            }
        }

        // Словарь
        $vocabulary = [
            'ноутбук', 'телефон', 'планшет',
            'стоимость', 'цена', 'дорого', 'дешево', 'бюджет',
            'наличие', 'есть', 'нет', 'склад', 'заказать', 'купить',
        ];

        // Обучающие данные
        $training_data = [
            ['текст' => 'ноутбук', 'продукт' => 'ноутбук', 'запрос' => 'общее'],
            ['текст' => 'телефон', 'продукт' => 'телефон', 'запрос' => 'общее'],
            ['текст' => 'планшет', 'продукт' => 'планшет', 'запрос' => 'общее'],

            ['текст' => 'стоимость ноутбука', 'продукт' => 'ноутбук', 'запрос' => 'стоимость'],
            ['текст' => 'цена на телефон', 'продукт' => 'телефон', 'запрос' => 'стоимость'],
            ['текст' => 'сколько стоит планшет', 'продукт' => 'планшет', 'запрос' => 'стоимость'],

            ['текст' => 'наличие ноутбуков', 'продукт' => 'ноутбук', 'запрос' => 'наличие'],
            ['текст' => 'есть телефон', 'продукт' => 'телефон', 'запрос' => 'наличие'],
            ['текст' => 'планшет в наличии', 'продукт' => 'планшет', 'запрос' => 'наличие'],

            ['текст' => 'ноутбук купить', 'продукт' => 'ноутбук', 'запрос' => 'общее'],
            ['текст' => 'дешевый телефон', 'продукт' => 'телефон', 'запрос' => 'стоимость'],
            ['текст' => 'планшет на складе', 'продукт' => 'планшет', 'запрос' => 'наличие'],
        ];


        // Преобразование данных
        $inputs = [];
        $outputs = [];

        foreach($training_data as $item)
        {
            $input_vector = $this->textToVector($item['текст'], $vocabulary);

            // Создаем комбинированный выходной вектор
            $category_index = array_search($item['продукт'].'_'.$item['запрос'], $all_categories);
            $output_vector = array_fill(0, count($all_categories), 0);
            $output_vector[$category_index] = 1;

            $inputs[] = $input_vector;
            $outputs[] = $output_vector;
        }

        //dd($inputs);  /* TODO: удалить !!! */
        //dd($outputs); /* TODO: удалить !!! */


        // Инициализация весов
        $input_neurons = count($vocabulary);
        $output_neurons = count($all_categories);

        $w1 = $brain->getRandMatrix($input_neurons, $hidden_layer_neurons);
        $w2 = $brain->getRandMatrix($hidden_layer_neurons, $output_neurons);
        $b1 = $brain->getRandMatrix($hidden_layer_neurons, 1);
        $b2 = $brain->getRandMatrix($output_neurons, 1);


        // Обучение модели
        for($i = 0; $i < $epochs; $i++)
        {
            foreach($inputs as $index => $inp)
            {
                $input_col = $brain->arrayTranspose($inp);
                $output_col = $brain->arrayTranspose($outputs[$index]);

                // Прямое распространение
                $forward = $brain->forward($input_col, $w1, $b1, $w2, $b2);

                // Обратное распространение
                $new_weights = $brain->backPropagation(
                    $forward,
                    $input_col,
                    $output_col,
                    $w1, $w2, $b1, $b2,
                );

                $w1 = $new_weights['w1'];
                $w2 = $new_weights['w2'];
                $b1 = $new_weights['b1'];
                $b2 = $new_weights['b2'];
            }
        }


        // Тестирование модели
        $test_phrases = [
            'ноутбук',
            'телефон',
            'планшет',
            'цена ноутбука',
            'стоимость телефона',
            'сколько стоит планшет',
            'есть ноутбук',
            'наличие телефона',
            'планшет в наличии',
            'бюджетный телефон',
        ];


        echo "<h2>Результаты классификации:</h2>";

        foreach($test_phrases as $phrase)
        {
            $vector = $this->textToVector($phrase, $vocabulary);
            $input_col = $brain->arrayTranspose($vector);

            $prediction = $brain->forward($input_col, $w1, $b1, $w2, $b2);
            $result = $prediction['A'];

            // Преобразуем результат в плоский массив
            $flat_result = [];
            foreach($result as $row)
            {
                $flat_result = array_merge($flat_result, $row);
            }

            // Находим лучшую категорию
            $best_index = array_search(max($flat_result), $flat_result);
            $best_category = $all_categories[$best_index];

            // Разбиваем категорию на составляющие
            [$product, $request_type] = explode('_', $best_category);

            // Готовим ответы
            $responses = [
                'стоимость' => [
                    'ноутбук' => 'Стоимость ноутбуков: от 20 000 до 150 000 руб.',
                    'телефон' => 'Цены на телефоны: от 5 000 до 100 000 руб.',
                    'планшет' => 'Планшеты доступны по цене от 10 000 до 80 000 руб.',
                ],
                'наличие' => [
                    'ноутбук' => 'Ноутбуки в наличии: 15 моделей на складе',
                    'телефон' => 'Телефоны доступны: 25 моделей в наличии',
                    'планшет' => 'Планшеты на складе: 10 моделей готовы к отгрузке',
                ],
                'общее' => [
                    'ноутбук' => 'Ноутбуки: мощные устройства для работы и игр',
                    'телефон' => 'Смартфоны: лучшие модели с отличной камерой',
                    'планшет' => 'Планшеты: идеальные устройства для чтения и серфинга',
                ],
            ];

            // Формируем ответ
            $response = $responses[$request_type][$product];

            echo "$phrase → Продукт: $product, Запрос: $request_type".PHP_EOL;
            echo "Ответ: $response".PHP_EOL;

            // Дополнительная информация о вероятностях
            $probs = array_map(fn($v) => round($v * 100, 1), $flat_result);
            //echo "Вероятности: ".json_encode(array_combine($all_categories, $probs), JSON_PRETTY_PRINT).PHP_EOL.PHP_EOL;

            echo PHP_EOL;
        }


        return Command::SUCCESS;
    }


    protected function execute(InputInterface $input, OutputInterface $output): int
    {
        $io = new SymfonyStyle($input, $output);


        // Параметры нейронной сети
        $epochs = 30000; // 20000;
        $learning_rate = 0.05;
        $hidden_layer_neurons = 8;
        $activation_fun = 'relu'; // ['relu', 'sigmoid', 'tanh']


        $brain = new Brainy()
            ->setLearningRate($learning_rate)
            ->setActivationFun($activation_fun);


        // Словарь и категории
        $categories = [
            'ноутбук',
            'телефон',
            'планшет',
            'стоимость',
            'наличие',
        ];


        $vocabulary = [
            'ноутбук', 'телефон', 'планшет', 'купить', 'продать', 'стоимость', 'наличие',
            'нужен', 'хочу', 'срочно', 'новый', 'мощный',
        ];


        // Подготовка данных
        $training_data = [
            ['текст' => 'ноутбук', 'категория' => 'ноутбук'],
            ['текст' => 'телефон', 'категория' => 'телефон'],
            ['текст' => 'планшет', 'категория' => 'планшет'],


            ['текст' => 'купить ноутбук', 'категория' => 'ноутбук'],
            ['текст' => 'хочу мощный телефон', 'категория' => 'ноутбук'],


            ['текст' => 'продать телефон', 'категория' => 'телефон'],
            ['текст' => 'наличие телефон', 'категория' => 'телефон'],

            ['текст' => 'купить планшет', 'категория' => 'планшет'],
            ['текст' => 'стоимость планшет', 'категория' => 'планшет'],

        ];


        // Преобразование данных
        $inputs = [];
        $outputs = [];

        foreach($training_data as $item)
        {
            $input_vector = $this->textToVector($item['текст'], $vocabulary);
            $output_vector = array_map(fn($cat) => (int) ($cat === $item['категория']), $categories);

            $inputs[] = $input_vector;
            $outputs[] = $output_vector;
        }

        // Инициализация весов
        $input_neurons = count($vocabulary);
        $output_neurons = count($categories);

        $w1 = $brain->getRandMatrix($input_neurons, $hidden_layer_neurons);
        $w2 = $brain->getRandMatrix($hidden_layer_neurons, $output_neurons);
        $b1 = $brain->getRandMatrix($hidden_layer_neurons, 1);
        $b2 = $brain->getRandMatrix($output_neurons, 1);

        // Обучение модели
        for($i = 0; $i < $epochs; $i++)
        {
            foreach($inputs as $index => $input)
            {
                $input_col = $brain->arrayTranspose($input);
                $output_col = $brain->arrayTranspose($outputs[$index]);

                // Прямое распространение
                $forward = $brain->forward($input_col, $w1, $b1, $w2, $b2);

                // Обратное распространение
                $new_weights = $brain->backPropagation(
                    $forward,
                    $input_col,
                    $output_col,
                    $w1, $w2, $b1, $b2,
                );

                $w1 = $new_weights['w1'];
                $w2 = $new_weights['w2'];
                $b1 = $new_weights['b1'];
                $b2 = $new_weights['b2'];
            }
        }

        // Тестирование модели
        $test_phrases = [
            'ноутбук',
            'телефон',
            'планшет',
            'купить ноутбук',
            'продать планшет',
            'мне нужен телефон',
            'какой телефон посоветуйте',
            'игровой ноутбук',
            'игровой телефон',
            'сколько в наличии телефон',
            'сколько стоит телефон',
        ];


        echo "<h2>Результаты классификации:</h2>".PHP_EOL;

        foreach($test_phrases as $phrase)
        {
            $vector = $this->textToVector($phrase, $vocabulary);
            $input_col = $brain->arrayTranspose($vector);

            $prediction = $brain->forward($input_col, $w1, $b1, $w2, $b2);
            $result = $prediction['A'];

            // Преобразуем результат в плоский массив
            $flat_result = [];
            foreach($result as $row)
            {
                $flat_result = array_merge($flat_result, $row);
            }

            $category_index = array_search(max($flat_result), $flat_result);
            $category = $categories[$category_index];

            // Ответы для категорий
            $responses = [
                'ноутбук' => 'Ноутбуки: мощные устройства для работы и игр',
                'телефон' => 'Смартфоны: лучшие модели с отличной камерой',
                'планшет' => 'Планшеты: идеальные устройства для чтения и серфинга',
            ];


            echo $phrase.PHP_EOL;
            echo "( контекст $category ): {$responses[$category]}".PHP_EOL;
            echo "Проценты: ".json_encode($flat_result).PHP_EOL.PHP_EOL;
        }

        dd('----------------------------------------------------------'); /* TODO: удалить !!! */


        $io->success('baks:machine:learning');

        return Command::SUCCESS;
    }


    /**
     * Функция, которая преобразует строку в числовой вектор
     */
    function textToVector($text, $vocabulary)
    {

        $vector = array_fill(0, count($vocabulary), 0);
        $words = explode(' ', mb_strtolower($text));

        foreach($words as $word)
        {
            // Поиск частичных совпадений
            foreach($vocabulary as $index => $vocabWord)
            {
                if(strpos($word, $vocabWord) !== false)
                {
                    $vector[$index] = 1;
                }
            }
        }

        return $vector;


        //        /** Текст поиска */
        //        $text = strtolower($text);
        //        $words = explode(' ', $text);
        //
        //        /** Строим вектор согласно словарному запасу */
        //        $vector = array_fill(0, count($vocabulary), 0);
        //
        //        /** Присваиваем вектору индекс слов */
        //        foreach($words as $word)
        //        {
        //            if(($index = array_search($word, $vocabulary)) !== false)
        //            {
        //                $vector[$index] = 1;
        //            }
        //        }
        //
        //        return $vector;
    }


}