#!/usr/bin/env python3
"""
Анализ корреляции между средней позицией товара и количеством заказов
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')



# Статические даты для примера
START_DATE = "2026-01-01"
END_DATE = "2026-02-01"

def get_advertising_data(db_name, ad_table, start_date, end_date):
    """
    Получение данных рекламы из базы данных
    """
    print(f"Загрузка данных из {db_name} за период {start_date} - {end_date}")
    
    try:
        conn = sqlite3.connect(db_name)
        
        query = f"""
        SELECT * FROM {ad_table} 
        WHERE strftime('%Y-%m-%d', date) BETWEEN ? AND ?
        """
        
        df = pd.read_sql_query(query, conn, params=(start_date, end_date))
        conn.close()
        
        if df.empty:
            print("Нет данных рекламы для указанного периода")
            return pd.DataFrame()
        
        print(f"Загружено {len(df)} записей")
        return df
        
    except sqlite3.Error as e:
        print(f"Ошибка SQLite: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Общая ошибка: {e}")
        return pd.DataFrame()

def analyze_position_orders_correlation(df):
    """
    Анализ корреляции между средней позицией и количеством заказов
    """
    if df.empty:
        print("Нет данных для анализа")
        return None
    
    # Проверяем наличие нужных колонок
    required_columns = ['avg_pos', 'orders', 'norm_query', 'advert_id', 'nm_id']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Отсутствуют колонки: {missing_columns}")
        return None
    
    print("\n" + "="*60)
    print("АНАЛИЗ КОРРЕЛЯЦИИ МЕЖДУ СРЕДНЕЙ ПОЗИЦИЕЙ И КОЛИЧЕСТВОМ ЗАКАЗОВ")
    print("="*60)
    
    # 1. Предварительная обработка данных
    df_clean = df.copy()
    
    # Убираем строки с NaN в ключевых колонках
    df_clean = df_clean.dropna(subset=['avg_pos', 'orders'])
    
    # Преобразуем типы данных
    df_clean['avg_pos'] = pd.to_numeric(df_clean['avg_pos'], errors='coerce')
    df_clean['orders'] = pd.to_numeric(df_clean['orders'], errors='coerce')
    
    # УБИРАЕМ ЗАКАЗЫ РАВНЫЕ НУЛЮ - ОЧИСТКА ДАННЫХ
    original_count = len(df_clean)
    df_clean = df_clean[df_clean['orders'] > 0]
    removed_zero_orders = original_count - len(df_clean)
    
    print(f"Удалено записей с 0 заказами: {removed_zero_orders}")
    print(f"Осталось записей для анализа: {len(df_clean)}")
    
    if len(df_clean) < 10:
        print("⚠️  Слишком мало данных для анализа после очистки")
        return None
    
    # Убираем выбросы в позиции (слишком большие значения)
    df_clean = df_clean[df_clean['avg_pos'] <= 200]  # Позиции больше 200 считаем выбросами
    df_clean = df_clean[df_clean['avg_pos'] > 0]  # Позиции должны быть положительными
    
    print(f"Анализируем {len(df_clean)} записей после очистки")
    
    # 2. Общая статистика
    print("\n📊 ОБЩАЯ СТАТИСТИКА:")
    print(f"Средняя позиция: {df_clean['avg_pos'].mean():.2f}")
    print(f"Медианная позиция: {df_clean['avg_pos'].median():.2f}")
    print(f"Минимальная позиция: {df_clean['avg_pos'].min():.2f}")
    print(f"Максимальная позиция: {df_clean['avg_pos'].max():.2f}")
    print(f"Среднее количество заказов: {df_clean['orders'].mean():.2f}")
    print(f"Медианное количество заказов: {df_clean['orders'].median():.2f}")
    print(f"Общее количество заказов: {df_clean['orders'].sum():.0f}")
    
    # 3. Группировка по позициям
    print("\n📈 АНАЛИЗ ПО ГРУППАМ ПОЗИЦИЙ:")
    
    # Создаем группы позиций
    bins = [0, 10, 20, 30, 50, 100, 200]
    labels = ['Топ-10', '11-20', '21-30', '31-50', '51-100', '100+']
    
    df_clean['position_group'] = pd.cut(df_clean['avg_pos'], bins=bins, labels=labels, right=False)
    
    group_stats = df_clean.groupby('position_group').agg({
        'orders': ['count', 'mean', 'sum', 'median'],
        'avg_pos': 'mean'
    }).round(2)
    
    print(group_stats)
    
    # 4. Расчет корреляции
    print("\n🔗 РАСЧЕТ КОРРЕЛЯЦИИ:")
    
    try:
        # Разные типы корреляции
        pearson_corr, pearson_p = stats.pearsonr(df_clean['avg_pos'], df_clean['orders'])
        spearman_corr, spearman_p = stats.spearmanr(df_clean['avg_pos'], df_clean['orders'])
        
        print(f"Коэффициент корреляции Пирсона: {pearson_corr:.4f}")
        print(f"p-значение Пирсона: {pearson_p:.6f}")
        
        print(f"Коэффициент корреляции Спирмена: {spearman_corr:.4f}")
        print(f"p-значение Спирмена: {spearman_p:.6f}")
        
    except Exception as e:
        print(f"Ошибка при расчете корреляции: {e}")
        # Устанавливаем значения по умолчанию
        pearson_corr, pearson_p, spearman_corr, spearman_p = 0, 1, 0, 1
    
    # Интерпретация корреляции
    print("\n📋 ИНТЕРПРЕТАЦИЯ КОРРЕЛЯЦИИ:")
    
    def interpret_correlation(corr_value):
        abs_corr = abs(corr_value)
        if abs_corr >= 0.7:
            return "сильная"
        elif abs_corr >= 0.3:
            return "умеренная"
        elif abs_corr >= 0.1:
            return "слабая"
        else:
            return "очень слабая или отсутствует"
    
    direction = "отрицательная" if pearson_corr < 0 else "положительная"
    strength = interpret_correlation(abs(pearson_corr))
    
    print(f"Наблюдается {strength} {direction} корреляция между позицией и заказами")
    if pearson_corr < -0.3:
        print("📉 Чем ниже позиция (ближе к 1), тем больше заказов")
    elif pearson_corr > 0.3:
        print("📈 Чем выше позиция (дальше от 1), тем больше заказов")
    else:
        print("📊 Нет явной линейной зависимости")
    
    # 5. Анализ по отдельным кампаниям/артикулам
    print("\n🎯 АНАЛИЗ ПО ОТДЕЛЬНЫМ КАМПАНИЯМ:")
    
    # Берем топ-10 кампаний по количеству дней показов или заказов
    if 'date' in df_clean.columns:
        top_campaigns = df_clean.groupby(['advert_id', 'norm_query']).agg({
            'date': 'nunique',
            'avg_pos': 'mean',
            'orders': 'sum'
        }).nlargest(10, 'orders').reset_index()  # Используем orders для сортировки
        sort_by = "заказам"
    else:
        top_campaigns = df_clean.groupby(['advert_id', 'norm_query']).agg({
            'avg_pos': 'mean',
            'orders': 'sum'
        }).nlargest(10, 'orders').reset_index()
        sort_by = "заказам"
    
    print(f"Топ-10 кампаний по {sort_by}:")
    for idx, row in top_campaigns.iterrows():
        query_display = row['norm_query'][:30] + "..." if len(str(row['norm_query'])) > 30 else row['norm_query']
        print(f"  {idx+1}. Кампания {row['advert_id']} ({query_display}): "
              f"Позиция={row['avg_pos']:.1f}, Заказы={row['orders']}")
    
    # 6. Анализ лучших и худших позиций
    print("\n🏆 АНАЛИЗ ЛУЧШИХ И ХУДШИХ ПОЗИЦИЙ:")
    
    # Лучшие позиции (топ-5 по заказам)
    best_positions = df_clean.nsmallest(20, 'avg_pos').nlargest(5, 'orders')
    print("Лучшие комбинации позиция/заказы (низкая позиция + много заказов):")
    for idx, row in best_positions.iterrows():
        query_display = str(row['norm_query'])[:20] + "..." if len(str(row['norm_query'])) > 20 else row['norm_query']
        print(f"  Позиция {row['avg_pos']:.1f}: {row['orders']} заказов "
              f"({query_display})")
    
    # Худшие позиции (высокие позиции с малым количеством заказов)
    high_pos_low_orders = df_clean[df_clean['avg_pos'] > 30].nsmallest(5, 'orders')
    if len(high_pos_low_orders) > 0:
        print("\n❌ Худшие комбинации (высокая позиция + мало заказов):")
        for idx, row in high_pos_low_orders.iterrows():
            query_display = str(row['norm_query'])[:20] + "..." if len(str(row['norm_query'])) > 20 else row['norm_query']
            print(f"  Позиция {row['avg_pos']:.1f}: {row['orders']} заказов "
                  f"({query_display})")
    
    # 7. Рекомендации
    print("\n💡 РЕКОМЕНДАЦИИ НА ОСНОВЕ АНАЛИЗА:")
    
    if pearson_corr < -0.3:
        print("1. 💎 Улучшение позиции существенно влияет на заказы")
        print("2. 📊 Сосредоточьтесь на выводе товаров в топ-20 позиций")
        print("3. 🎯 Оптимизируйте кампании с позицией 21-50 для роста")
        print("4. ⭐ Цель: снизить среднюю позицию до 20 или ниже")
    elif abs(pearson_corr) < 0.3:
        print("1. 🔍 Нет сильной линейной зависимости позиции и заказов")
        print("2. 📈 Другие факторы (CTR, конверсия, цена) могут быть важнее")
        print("3. 🎪 Проверьте качество трафика по разным позициям")
        print("4. 🔎 Изучите конкретные кампании с хорошими результатами")
    
    # Общие рекомендации
    avg_position = df_clean['avg_pos'].mean()
    if avg_position > 30:
        print(f"5. ⬆️ Средняя позиция {avg_position:.1f} - есть потенциал для роста")
        print(f"   Цель: снизить среднюю позицию до 25")
    elif avg_position < 20:
        print(f"5. ✅ Отличная средняя позиция: {avg_position:.1f}")
        print(f"   Поддерживайте текущие результаты")
    
    # Анализ эффективности разных групп позиций
    if 'position_group' in df_clean.columns:
        top10_data = df_clean[df_clean['position_group'] == 'Топ-10']
        if not top10_data.empty:
            top10_efficiency = top10_data['orders'].sum() / len(top10_data)
            other_efficiency = df_clean[df_clean['position_group'] != 'Топ-10']['orders'].sum() / len(df_clean[df_clean['position_group'] != 'Топ-10'])
            
            if top10_efficiency > other_efficiency * 1.5:
                print(f"6. ⭐ Товары в топ-10 приносят в {top10_efficiency/other_efficiency:.1f} раз больше заказов на запись")
                print(f"   Увеличивайте бюджет на топовые позиции")
    
    # 8. Визуализация
    create_visualizations(df_clean, pearson_corr)
    
    return {
        'data': df_clean,
        'pearson_corr': pearson_corr,
        'pearson_p': pearson_p,
        'spearman_corr': spearman_corr,
        'spearman_p': spearman_p,
        'group_stats': group_stats,
        'top_campaigns': top_campaigns
    }

def create_visualizations(df, correlation):
    """
    Создание визуализаций для анализа
    """
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Анализ корреляции позиции и заказов (корреляция: {correlation:.3f})', 
                fontsize=16, fontweight='bold')
    
    # 1. Точечный график корреляции
    ax1 = axes[0, 0]
    scatter = ax1.scatter(df['avg_pos'], df['orders'], alpha=0.6, c='blue', edgecolors='black', linewidth=0.5, s=30)
    ax1.set_xlabel('Средняя позиция (avg_pos)')
    ax1.set_ylabel('Количество заказов')
    ax1.set_title('Корреляция позиции и заказов')
    ax1.grid(True, alpha=0.3)
    
    # Линия тренда
    if len(df) > 1:
        try:
            z = np.polyfit(df['avg_pos'], df['orders'], 1)
            p = np.poly1d(z)
            ax1.plot(df['avg_pos'], p(df['avg_pos']), "r--", alpha=0.8, linewidth=2)
        except:
            pass
    
    # 2. Распределение позиций
    ax2 = axes[0, 1]
    ax2.hist(df['avg_pos'], bins=30, edgecolor='black', alpha=0.7, color='skyblue')
    ax2.set_xlabel('Средняя позиция')
    ax2.set_ylabel('Частота')
    ax2.set_title('Распределение позиций')
    ax2.grid(True, alpha=0.3)
    
    # 3. Распределение заказов
    ax3 = axes[0, 2]
    # Логарифмическая шкала для лучшей визуализации
    orders_log = np.log1p(df['orders'])
    ax3.hist(orders_log, bins=30, edgecolor='black', alpha=0.7, color='lightgreen')
    ax3.set_xlabel('log(Заказы + 1)')
    ax3.set_ylabel('Частота')
    ax3.set_title('Распределение заказов (логарифм)')
    ax3.grid(True, alpha=0.3)
    
    # 4. Средние заказы по группам позиций
    ax4 = axes[1, 0]
    if 'position_group' in df.columns:
        group_means = df.groupby('position_group')['orders'].mean()
        group_counts = df.groupby('position_group').size()
        
        # Создаем DataFrame для сортировки
        group_data = pd.DataFrame({
            'mean_orders': group_means,
            'count': group_counts
        }).dropna()
        
        if not group_data.empty:
            # Сортируем по количеству записей
            group_data = group_data.sort_values('count', ascending=False)
            
            colors = plt.cm.Blues(np.linspace(0.5, 0.9, len(group_data)))
            bars = ax4.bar(group_data.index.astype(str), group_data['mean_orders'].values, 
                          color=colors, edgecolor='black')
            ax4.set_xlabel('Группа позиций')
            ax4.set_ylabel('Средние заказы')
            ax4.set_title('Средние заказы по группам позиций')
            ax4.tick_params(axis='x', rotation=45)
            
            # Добавляем значения на столбцы
            for i, (idx, row) in enumerate(group_data.iterrows()):
                ax4.text(i, row['mean_orders'] + 0.1, f'{row["mean_orders"]:.1f}', 
                        ha='center', va='bottom', fontsize=9)
                # Добавляем количество записей под столбцом
                ax4.text(i, -max(group_data['mean_orders']) * 0.05, f'n={row["count"]}', 
                        ha='center', va='top', fontsize=8, color='gray')
    
    # 5. Box plot по группам позиций
    ax5 = axes[1, 1]
    if 'position_group' in df.columns:
        # Используем группы с достаточным количеством данных
        valid_groups = df['position_group'].value_counts()
        valid_groups = valid_groups[valid_groups >= 5].index.tolist()
        
        if len(valid_groups) >= 2:
            data_to_plot = [df[df['position_group'] == group]['orders'] for group in valid_groups[:4]]
            box = ax5.boxplot(data_to_plot, labels=valid_groups[:4], patch_artist=True)
            
            # Цвета для box plot
            colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightsalmon']
            for patch, color in zip(box['boxes'], colors[:len(data_to_plot)]):
                patch.set_facecolor(color)
            
            ax5.set_xlabel('Группа позиций')
            ax5.set_ylabel('Заказы')
            ax5.set_title('Распределение заказов по группам позиций')
            ax5.grid(True, alpha=0.3, axis='y')
        else:
            ax5.text(0.5, 0.5, 'Недостаточно данных\nдля box plot', 
                    ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('Недостаточно данных')
    
    # 6. Heatmap корреляции (если есть другие параметры)
    ax6 = axes[1, 2]
    # Выбираем числовые колонки для корреляционной матрицы
    numeric_cols = ['avg_pos', 'orders']
    additional_cols = ['views', 'clicks', 'cpc', 'atbs']
    
    for col in additional_cols:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
    
    if len(numeric_cols) > 2:
        try:
            corr_matrix = df[numeric_cols].corr()
            im = ax6.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            ax6.set_xticks(range(len(numeric_cols)))
            ax6.set_yticks(range(len(numeric_cols)))
            ax6.set_xticklabels(numeric_cols, rotation=45, ha='right')
            ax6.set_yticklabels(numeric_cols)
            ax6.set_title('Корреляционная матрица')
            
            # Добавляем значения в ячейки
            for i in range(len(numeric_cols)):
                for j in range(len(numeric_cols)):
                    text = ax6.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                   ha="center", va="center", color="black", fontsize=9)
            
            plt.colorbar(im, ax=ax6)
        except:
            # Если не удалось создать heatmap, показываем информацию о данных
            ax6.text(0.5, 0.5, f'Данные: {len(df)} записей\nКорреляция: {correlation:.3f}', 
                    ha='center', va='center', transform=ax6.transAxes, fontsize=12)
            ax6.set_title('Информация о данных')
            ax6.axis('off')
    else:
        # Если нет других числовых колонок, показываем QQ-plot
        try:
            from scipy import stats
            stats.probplot(df['orders'], dist="norm", plot=ax6)
            ax6.set_title('QQ-plot распределения заказов')
            ax6.grid(True, alpha=0.3)
        except:
            ax6.text(0.5, 0.5, 'Не удалось создать QQ-plot', 
                    ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title('Ошибка визуализации')
    
    plt.tight_layout()
    
    # Сохраняем график
    filename = f'position_orders_correlation_{START_DATE}_to_{END_DATE}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\n📊 Графики сохранены в файл: {filename}")
    
    # Создаем дополнительный график: тренд заказов по позициям
    plt.figure(figsize=(10, 6))
    
    # Группируем по позициям (округляем для группировки)
    df['pos_rounded'] = df['avg_pos'].round()
    trend_data = df.groupby('pos_rounded').agg({
        'orders': ['mean', 'count', 'std']
    }).reset_index()
    
    trend_data.columns = ['position', 'avg_orders', 'count', 'std_orders']
    
    # Фильтруем позиции с достаточным количеством данных
    trend_data = trend_data[trend_data['count'] >= 3]
    
    if not trend_data.empty:
        plt.errorbar(trend_data['position'], trend_data['avg_orders'], 
                    yerr=trend_data['std_orders'], 
                    fmt='o-', capsize=5, capthick=2, 
                    markersize=6, linewidth=2, alpha=0.8)
        
        plt.xlabel('Позиция (округленная)')
        plt.ylabel('Средние заказы')
        plt.title('Тренд заказов по позициям')
        plt.grid(True, alpha=0.3)
        plt.xlim(0, min(100, trend_data['position'].max() + 5))
        
        # Сохраняем второй график
        trend_filename = f'position_trend_{START_DATE}_to_{END_DATE}.png'
        plt.savefig(trend_filename, dpi=150, bbox_inches='tight')
        print(f"📈 График тренда сохранен в файл: {trend_filename}")
    else:
        print("⚠️  Недостаточно данных для создания графика тренда")
    
    plt.show()

def save_results_to_excel(results, filename=None):
    """
    Сохранение результатов анализа в Excel
    """
    if results is None:
        print("⚠️  Нет результатов для сохранения")
        return
    
    if filename is None:
        filename = f'position_orders_analysis_{START_DATE}_to_{END_DATE}.xlsx'
    
    try:
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Основные данные
            results['data'].to_excel(writer, sheet_name='Данные', index=False)
            
            # Статистика по группам
            results['group_stats'].to_excel(writer, sheet_name='Групповая статистика')
            
            # Топ кампании
            results['top_campaigns'].to_excel(writer, sheet_name='Топ кампании', index=False)
            
            # Сводная статистика
            summary_data = {
                'Метрика': [
                    'Коэффициент корреляции Пирсона', 
                    'p-значение Пирсона',
                    'Коэффициент корреляции Спирмена',
                    'p-значение Спирмена',
                    'Количество записей',
                    'Средняя позиция',
                    'Медианная позиция',
                    'Средние заказы',
                    'Медианные заказы',
                    'Общее количество заказов'
                ],
                'Значение': [
                    results.get('pearson_corr', 0),
                    results.get('pearson_p', 0),
                    results.get('spearman_corr', 0),
                    results.get('spearman_p', 0),
                    len(results['data']),
                    results['data']['avg_pos'].mean(),
                    results['data']['avg_pos'].median(),
                    results['data']['orders'].mean(),
                    results['data']['orders'].median(),
                    results['data']['orders'].sum()
                ]
            }
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Сводка', index=False)
            
            # Дополнительный анализ: топ-20 записей по заказам
            top_20_by_orders = results['data'].nlargest(20, 'orders')[['norm_query', 'advert_id', 'nm_id', 'avg_pos', 'orders']]
            top_20_by_orders.to_excel(writer, sheet_name='Топ-20 по заказам', index=False)
            
            # Дополнительный анализ: топ-20 лучших позиций
            top_20_by_position = results['data'].nsmallest(20, 'avg_pos')[['norm_query', 'advert_id', 'nm_id', 'avg_pos', 'orders']]
            top_20_by_position.to_excel(writer, sheet_name='Топ-20 позиций', index=False)
        
        print(f"\n💾 Результаты сохранены в Excel файл: {filename}")
        
    except Exception as e:
        print(f"Ошибка при сохранении в Excel: {e}")

def main():
    """
    Основная функция программы
    """
    print("="*70)
    print("АНАЛИЗ КОРРЕЛЯЦИИ МЕЖДУ СРЕДНЕЙ ПОЗИЦИЕЙ И КОЛИЧЕСТВОМ ЗАКАЗОВ")
    print("="*70)
    print(f"Период анализа: {START_DATE} - {END_DATE}")
    print(f"База данных: {DB_NAME}")
    print(f"Таблица: {AD_TABLE}")
    print("="*70)
    
    # Загружаем данные
    df = get_advertising_data(DB_NAME, AD_TABLE, START_DATE, END_DATE)
    
    if df.empty:
        print("Не удалось загрузить данные. Проверьте настройки подключения.")
        return
    
    # Проверяем наличие необходимых колонок
    if 'avg_pos' not in df.columns or 'orders' not in df.columns:
        print("В данных отсутствуют необходимые колонки 'avg_pos' или 'orders'")
        print(f"Доступные колонки: {list(df.columns)}")
        return
    
    # Анализируем корреляцию
    results = analyze_position_orders_correlation(df)
    
    if results:
        # Сохраняем результаты в Excel
        save_results_to_excel(results)
        
        print("\n" + "="*70)
        print("АНАЛИЗ ЗАВЕРШЕН")
        print("="*70)
        
        # Выводим краткий итог
        print(f"\n📋 КРАТКИЕ ИТОГИ:")
        print(f"   • Корреляция Пирсона: {results['pearson_corr']:.4f}")
        
        if results['pearson_p'] < 0.05:
            significance = "статистически значима"
        else:
            significance = "не статистически значима"
        
        print(f"   • Статистическая значимость: {significance} (p={results['pearson_p']:.4f})")
        
        if results['pearson_corr'] < -0.3:
            print("   • Вывод: Позиция существенно влияет на заказы")
            print("   • Рекомендация: Улучшайте позиции в поиске")
        elif abs(results['pearson_corr']) < 0.3:
            print("   • Вывод: Нет сильной линейной зависимости")
            print("   • Рекомендация: Изучите другие факторы (CTR, конверсия)")
        else:
            print("   • Вывод: Неожиданная положительная корреляция")
            print("   • Рекомендация: Проверьте качество данных")
    else:
        print("Анализ не выполнен из-за проблем с данными")

if __name__ == "__main__":

    # Перед запуском замените эти значения на ваши реальные
  
    # DB_NAME = "ваша_база_данных.db"  # Замените на путь к вашей БД
    # AD_TABLE = "ваша_таблица_рекламы"  # Замените на имя вашей таблицы
    
    main()
