import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import io
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuração da página
st.set_page_config(
    page_title="Análise Estatística de Dados",
    page_icon="📊",
    layout="wide"
)

# Título da aplicação
st.title("📊 Análise Estatística - Dados 3° Séries e 9° Anos")

# Função para carregar dados
@st.cache_data
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        st.error(f"Erro ao carregar o arquivo {file_path}: {e}")
        return None

# Função para análise estatística básica
def basic_statistics(data, dataset_name):
    st.header(f"📈 Descrição dados - {dataset_name}")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Informações Gerais")
        st.write(f"Total de registros: {data.shape[0]}")
        st.write(f"Total de variáveis: {data.shape[1]}")
        st.write(f"Valores nulos: {data.isnull().sum().sum()}")
    
    with col2:
        st.subheader("Tipos de Dados")
        dtype_counts = data.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            st.write(f"{dtype}: {count}")
    
    with col3:
        st.subheader("Amostra dos Dados")
        st.dataframe(data.head(), use_container_width=True)

# Função para encontrar colunas similares
def find_similar_columns(data, keywords):
    """Encontra colunas que contenham as keywords fornecidas"""
    found_columns = []
    for col in data.columns:
        col_lower = col.lower()
        for keyword in keywords:
            if keyword.lower() in col_lower:
                found_columns.append(col)
                break
    return found_columns

# Função para análise estatística por escola
def school_statistics(data, dataset_name):
    st.header(f"🏫 Estatísticas por Escola - {dataset_name}")
    
    # Mostrar todas as colunas disponíveis para debug
    with st.expander("🔍 Ver todas as colunas disponíveis"):
        st.write(f"Total de colunas: {len(data.columns)}")
        st.write("Lista de colunas:")
        for i, col in enumerate(data.columns):
            st.write(f"{i+1}. {col} (tipo: {data[col].dtype})")
    
    # Encontrar colunas similares
    st.subheader("🎯 Seleção de Colunas para Análise")
    
    # Procurar por coluna de pontos
    pontos_keywords = ['total', 'pontos', 'points', 'score', 'nota']
    colunas_pontos = find_similar_columns(data, pontos_keywords)
    
    # Procurar por coluna de escola
    escola_keywords = ['escola', 'school', 'colégio', 'colegio', 'instituição']
    colunas_escola = find_similar_columns(data, escola_keywords)
    
    # Seleção manual de colunas se não encontrar automaticamente
    if not colunas_pontos or not colunas_escola:
        st.warning("⚠️ Não foi possível encontrar as colunas automaticamente. Selecione manualmente:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            coluna_total_pontos = st.selectbox(
                f"Selecione a coluna de Total de Pontos - {dataset_name}",
                data.columns,
                key=f"pontos_{dataset_name}"
            )
        
        with col2:
            coluna_escola = st.selectbox(
                f"Selecione a coluna de Escola - {dataset_name}",
                data.columns,
                key=f"escola_{dataset_name}"
            )
    else:
        # Usar as primeiras colunas encontradas
        coluna_total_pontos = colunas_pontos[0]
        coluna_escola = colunas_escola[0]
        
        st.success(f"✅ Coluna de pontos selecionada: **{coluna_total_pontos}**")
        st.success(f"✅ Coluna de escola selecionada: **{coluna_escola}**")
    
    # Preparar os dados
    with st.status("🔄 Preparando dados para análise...", expanded=True) as status:
        # Criar cópia dos dados relevantes
        data_analysis = data[[coluna_escola, coluna_total_pontos]].copy()
        
        # Converter coluna de pontos para numérico se necessário
        if data_analysis[coluna_total_pontos].dtype == 'object':
            st.write("Convertendo coluna de pontos para numérico...")
            data_analysis[coluna_total_pontos] = pd.to_numeric(
                data_analysis[coluna_total_pontos], 
                errors='coerce'
            )
        
        # Remover linhas com valores nulos
        initial_count = len(data_analysis)
        data_clean = data_analysis.dropna()
        final_count = len(data_clean)
        
        st.write(f"Registros antes da limpeza: {initial_count}")
        st.write(f"Registros após remover valores nulos: {final_count}")
        st.write(f"Registros removidos: {initial_count - final_count}")
        
        if final_count == 0:
            st.error("❌ Não há dados válidos para análise após a limpeza.")
            return
        
        # Verificar se há dados após a limpeza
        escolas_unicas = data_clean[coluna_escola].nunique()
        st.write(f"Número de escolas únicas: {escolas_unicas}")
        
        if escolas_unicas == 0:
            st.error("❌ Não há escolas para analisar.")
            return
        
        status.update(label="✅ Dados preparados com sucesso!", state="complete", expanded=False)
    
    # Calcular estatísticas por escola
    try:
        estatisticas_escolas = data_clean.groupby(coluna_escola)[coluna_total_pontos].agg([
            ('Média', 'mean'),
            ('Mediana', 'median'),
            ('Mínimo', 'min'),
            ('Máximo', 'max'),
            ('Desvio_Padrão', 'std'),
            ('Quantidade_Alunos', 'count')
        ]).round(2)
        
        # Ordenar por média (opcional)
        estatisticas_escolas = estatisticas_escolas.sort_values('Média', ascending=False)
        
        # Reset index para usar a coluna de escola nos gráficos
        estatisticas_escolas_reset = estatisticas_escolas.reset_index()
        
        # Mostrar tabela de estatísticas
        st.subheader("📋 Estatísticas por Escola")
        
        # Adicionar filtros para a tabela
        col1, col2 = st.columns(2)
        with col1:
            min_alunos = st.slider(
                "Filtrar por quantidade mínima de alunos",
                min_value=1,
                max_value=int(estatisticas_escolas['Quantidade_Alunos'].max()),
                value=1,
                key=f"min_alunos_{dataset_name}"
            )
        
        with col2:
            escolas_filtradas = st.multiselect(
                "Selecionar escolas específicas",
                options=estatisticas_escolas_reset[coluna_escola].tolist(),
                default=estatisticas_escolas_reset[coluna_escola].head(10).tolist(),
                key=f"escolas_select_{dataset_name}"
            )
        
        # Aplicar filtros
        estatisticas_filtradas = estatisticas_escolas_reset[
            (estatisticas_escolas_reset['Quantidade_Alunos'] >= min_alunos) &
            (estatisticas_escolas_reset[coluna_escola].isin(escolas_filtradas) if escolas_filtradas else True)
        ]
        
        st.dataframe(estatisticas_filtradas, use_container_width=True)
        
        # Gráficos interativos
        st.subheader("📊 Visualizações Interativas")
        
        # Layout com tabs para diferentes visualizações
        tab1, tab2, tab3, tab4 = st.tabs(["🏆 Médias por Escola", "👥 Quantidade de Alunos", "📈 Comparação Detalhada", "🎯 Distribuição"])
        
        with tab1:
            # Gráfico de médias por escola - INTERATIVO
            fig_medias = px.bar(
                estatisticas_filtradas.sort_values('Média', ascending=True),
                x='Média',
                y=coluna_escola,
                orientation='h',
                title=f'🏆 Média de Pontos por Escola - {dataset_name}',
                color='Média',
                color_continuous_scale='viridis',
                hover_data={
                    'Média': ':.2f',
                    'Mediana': ':.2f',
                    'Quantidade_Alunos': True,
                    'Desvio_Padrão': ':.2f'
                }
            )
            
            fig_medias.update_layout(
                yaxis_title='Escola',
                xaxis_title='Média de Pontos',
                height=600,
                showlegend=False
            )
            
            # Adicionar linha da média geral
            media_geral = data_clean[coluna_total_pontos].mean()
            fig_medias.add_vline(
                x=media_geral, 
                line_dash="dash", 
                line_color="red",
                annotation_text=f"Média Geral: {media_geral:.2f}"
            )
            
            st.plotly_chart(fig_medias, use_container_width=True)
        
        with tab2:
            # Gráfico de quantidade de alunos por escola - INTERATIVO
            fig_alunos = px.bar(
                estatisticas_filtradas.sort_values('Quantidade_Alunos', ascending=True),
                x='Quantidade_Alunos',
                y=coluna_escola,
                orientation='h',
                title=f'👥 Quantidade de Alunos por Escola - {dataset_name}',
                color='Quantidade_Alunos',
                color_continuous_scale='plasma',
                hover_data={
                    'Quantidade_Alunos': True,
                    'Média': ':.2f',
                    'Mediana': ':.2f'
                }
            )
            
            fig_alunos.update_layout(
                yaxis_title='Escola',
                xaxis_title='Número de Alunos',
                height=600,
                showlegend=False
            )
            
            st.plotly_chart(fig_alunos, use_container_width=True)
        
        with tab3:
            # Scatter plot comparando média vs quantidade de alunos - INTERATIVO
            fig_scatter = px.scatter(
                estatisticas_filtradas,
                x='Quantidade_Alunos',
                y='Média',
                size='Quantidade_Alunos',
                color='Média',
                hover_name=coluna_escola,
                title=f'📈 Relação entre Quantidade de Alunos e Média de Pontos - {dataset_name}',
                size_max=60,
                color_continuous_scale='rainbow',
                hover_data={
                    'Mediana': ':.2f',
                    'Desvio_Padrão': ':.2f',
                    'Mínimo': ':.2f',
                    'Máximo': ':.2f'
                }
            )
            
            fig_scatter.update_layout(
                xaxis_title='Quantidade de Alunos',
                yaxis_title='Média de Pontos',
                height=500
            )
            
            # Adicionar linhas de referência
            fig_scatter.add_hline(
                y=media_geral, 
                line_dash="dash", 
                line_color="red",
                annotation_text=f"Média Geral"
            )
            
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with tab4:
            # Box plot interativo da distribuição por escola
            # Selecionar top escolas para o box plot (para não sobrecarregar o gráfico)
            top_escolas = estatisticas_filtradas.nlargest(15, 'Quantidade_Alunos')[coluna_escola].tolist()
            data_top_escolas = data_clean[data_clean[coluna_escola].isin(top_escolas)]
            
            fig_box = px.box(
                data_top_escolas,
                x=coluna_escola,
                y=coluna_total_pontos,
                title=f'🎯 Distribuição de Pontos por Escola (Top 15 por quantidade de alunos) - {dataset_name}',
                color=coluna_escola,
                points="all"
            )
            
            fig_box.update_layout(
                xaxis_title='Escola',
                yaxis_title='Pontos',
                height=500,
                showlegend=False
            )
            
            fig_box.update_xaxes(tickangle=45)
            
            st.plotly_chart(fig_box, use_container_width=True)
        
        # Estatísticas resumidas em cards
        st.subheader("📊 Resumo Geral")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total de Escolas", 
                estatisticas_escolas.shape[0],
                f"{estatisticas_filtradas.shape[0]} filtradas"
            )
            st.metric("Média Geral", f"{data_clean[coluna_total_pontos].mean():.2f}")
        
        with col2:
            st.metric(
                "Total de Alunos", 
                estatisticas_escolas['Quantidade_Alunos'].sum(),
                f"{estatisticas_filtradas['Quantidade_Alunos'].sum()} filtrados"
            )
            st.metric("Mediana Geral", f"{data_clean[coluna_total_pontos].median():.2f}")
        
        with col3:
            st.metric("Maior Média", f"{estatisticas_escolas['Média'].max():.2f}")
            st.metric("Menor Média", f"{estatisticas_escolas['Média'].min():.2f}")
        
        with col4:
            st.metric("Desvio Padrão Geral", f"{data_clean[coluna_total_pontos].std():.2f}")
            st.metric("Amplitude Total", f"{data_clean[coluna_total_pontos].max() - data_clean[coluna_total_pontos].min():.2f}")
        
        # Download dos dados processados
        st.subheader("💾 Exportar Dados")
        csv = estatisticas_escolas_reset.to_csv(index=False)
        st.download_button(
            label="📥 Baixar estatísticas por escola em CSV",
            data=csv,
            file_name=f"estatisticas_escolas_{dataset_name.lower().replace(' ', '_')}.csv",
            mime="text/csv"
        )
            
    except Exception as e:
        st.error(f"❌ Erro ao calcular estatísticas: {e}")
        st.info("Isso pode ocorrer se não houver dados numéricos suficientes para análise.")

# Função para análise detalhada das colunas
def column_analysis(data, dataset_name):
    st.header(f"🔍 Análise por Coluna - {dataset_name}")
    
    # Seleção múltipla de colunas
    st.subheader("🎯 Seleção de Colunas para Análise")
    
    # Encontrar colunas similares automaticamente
    escola_keywords = ['escola', 'school', 'colégio', 'colegio', 'instituição']
    nota_lp_keywords = ['nota lp', 'nota_lp', 'portugues', 'português', 'lingua portuguesa']
    nota_mat_keywords = ['nota mat', 'nota_mat', 'matemática', 'matematica', 'math']
    
    colunas_escola = find_similar_columns(data, escola_keywords)
    colunas_nota_lp = find_similar_columns(data, nota_lp_keywords)
    colunas_nota_mat = find_similar_columns(data, nota_mat_keywords)
    
    # Seleção de colunas
    col1, col2 = st.columns(2)
    
    with col1:
        coluna_agrupamento = st.selectbox(
            f"Selecione a coluna para agrupamento - {dataset_name}",
            options=data.columns.tolist(),
            index=data.columns.tolist().index(colunas_escola[0]) if colunas_escola else 0,
            key=f"agrupamento_{dataset_name}"
        )
    
    with col2:
        colunas_analise = st.multiselect(
            f"Selecione as colunas para análise - {dataset_name}",
            options=data.columns.tolist(),
            default=colunas_nota_lp[:1] + colunas_nota_mat[:1] if colunas_nota_lp or colunas_nota_mat else [],
            key=f"analise_{dataset_name}"
        )
    
    if not coluna_agrupamento or not colunas_analise:
        st.warning("⚠️ Selecione pelo menos uma coluna de agrupamento e uma coluna para análise.")
        return
    
    # Verificar se temos a combinação específica de colunas
    tem_escola = any(keyword in coluna_agrupamento.lower() for keyword in escola_keywords)
    tem_nota_lp = any(any(keyword in col.lower() for keyword in nota_lp_keywords) for col in colunas_analise)
    tem_nota_mat = any(any(keyword in col.lower() for keyword in nota_mat_keywords) for col in colunas_analise)
    
    # Análise específica para combinação escola + notas
    if tem_escola and (tem_nota_lp or tem_nota_mat):
        st.subheader("🏫 Análise de Notas por Escola")
        
        # Preparar dados
        colunas_analise_selecionadas = [coluna_agrupamento] + colunas_analise
        data_analysis = data[colunas_analise_selecionadas].copy()
        
        # Converter colunas numéricas
        for col in colunas_analise:
            if data_analysis[col].dtype == 'object':
                data_analysis[col] = pd.to_numeric(data_analysis[col], errors='coerce')
        
        # Remover valores nulos
        data_clean = data_analysis.dropna()
        
        if len(data_clean) == 0:
            st.error("❌ Não há dados válidos para análise após a limpeza.")
            return
        
        # Calcular estatísticas agrupadas
        estatisticas_escolas = data_clean.groupby(coluna_agrupamento)[colunas_analise].agg(['mean', 'std', 'count']).round(2)
        
        # Renomear colunas para melhor visualização
        estatisticas_escolas.columns = ['_'.join(col).strip() for col in estatisticas_escolas.columns.values]
        estatisticas_escolas = estatisticas_escolas.reset_index()
        
        # Criar descrição personalizada baseada nas colunas selecionadas
        descricao_analise = ""
        if tem_nota_lp and not tem_nota_mat:
            descricao_analise = "médias em Português"
        elif tem_nota_mat and not tem_nota_lp:
            descricao_analise = "médias em Matemática"
        elif tem_nota_lp and tem_nota_mat:
            descricao_analise = "médias em Português e Matemática"
        
        st.write(f"**Descrição:** {descricao_analise} agrupadas por {coluna_agrupamento}")
        
        # Mostrar tabela de estatísticas
        st.subheader("📊 Estatísticas Agrupadas")
        st.dataframe(estatisticas_escolas, use_container_width=True)
        
        # Gráficos interativos
        st.subheader("📈 Visualizações")
        
        # Preparar dados para gráficos
        dados_melted = data_clean.melt(
            id_vars=[coluna_agrupamento],
            value_vars=colunas_analise,
            var_name='Disciplina',
            value_name='Nota'
        )
        
        # Criar tabs para diferentes visualizações
        tab1, tab2, tab3, tab4 = st.tabs(["🏆 Ranking de Médias", "📈 Comparação Horizontal", "🎯 Distribuição", "📋 Estatísticas"])
        
        with tab1:
            # GRÁFICO HIERÁRQUICO - Escolas no eixo Y, médias no eixo X
            medias_por_escola = dados_melted.groupby([coluna_agrupamento, 'Disciplina'])['Nota'].mean().reset_index()
            
            # ORDENAÇÃO CORRIGIDA: Maior média em cima, menor em baixo
            ordem_escolas = medias_por_escola.groupby(coluna_agrupamento)['Nota'].mean().sort_values(ascending=False).index
            
            fig_hierarquico = px.bar(
                medias_por_escola,
                x='Nota',
                y=coluna_agrupamento,
                color='Disciplina',
                orientation='h',  # Barras horizontais
                title=f'🏆 Ranking de Médias por Escola - {dataset_name}',
                hover_data={
                    'Nota': ':.2f',
                    coluna_agrupamento: True
                },
                category_orders={coluna_agrupamento: ordem_escolas}  # Ordenar escolas (maior → menor)
            )
            
            fig_hierarquico.update_layout(
                yaxis_title='Escola',
                xaxis_title='Média de Notas',
                height=600,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Adicionar linha da média geral
            media_geral = data_clean[colunas_analise].mean().mean()
            fig_hierarquico.add_vline(
                x=media_geral, 
                line_dash="dash", 
                line_color="red",
                annotation_text=f"Média Geral: {media_geral:.2f}",
                annotation_position="top right"
            )
            
            st.plotly_chart(fig_hierarquico, use_container_width=True)
            
            # Legenda informativa
            st.info("🏅 **Ranking:** As escolas estão ordenadas da **maior** para a **menor** média geral. A escola no topo tem a melhor performance.")
            
            # Mostrar top 3 e bottom 3 escolas
            st.subheader("🎯 Destaques do Ranking")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**🥇 Top 3 Melhores Escolas:**")
                top_3 = ordem_escolas[:3]
                for i, escola in enumerate(top_3, 1):
                    medal = ["🥇", "🥈", "🥉"][i-1]
                    media_geral_escola = medias_por_escola.groupby(coluna_agrupamento)['Nota'].mean().loc[escola]
                    st.write(f"{medal} **{escola}:** {media_geral_escola:.2f}")
            
            with col2:
                st.write("**📉 3 Escolas com Menor Desempenho:**")
                bottom_3 = ordem_escolas[-3:]
                for i, escola in enumerate(reversed(bottom_3), 1):
                    media_geral_escola = medias_por_escola.groupby(coluna_agrupamento)['Nota'].mean().loc[escola]
                    st.write(f"📉 **{escola}:** {media_geral_escola:.2f}")
        
        with tab2:
            # Gráfico de comparação horizontal entre disciplinas
            if len(colunas_analise) > 1:
                # Pivot table para ter disciplinas como colunas
                pivot_medias = medias_por_escola.pivot(
                    index=coluna_agrupamento, 
                    columns='Disciplina', 
                    values='Nota'
                ).reset_index()
                
                # Ordenar pela média geral (MAIOR → MENOR)
                pivot_medias['Média_Geral'] = pivot_medias[colunas_analise].mean(axis=1)
                pivot_medias = pivot_medias.sort_values('Média_Geral', ascending=False)
                
                fig_comparacao = go.Figure()
                
                # Adicionar uma trace para cada disciplina
                colors = px.colors.qualitative.Set3
                for i, disciplina in enumerate(colunas_analise):
                    fig_comparacao.add_trace(go.Bar(
                        y=pivot_medias[coluna_agrupamento],
                        x=pivot_medias[disciplina],
                        name=disciplina,
                        orientation='h',
                        marker_color=colors[i % len(colors)],
                        hovertemplate=f'<b>%{{y}}</b><br>{disciplina}: %{{x:.2f}}<extra></extra>'
                    ))
                
                fig_comparacao.update_layout(
                    title=f'📊 Comparação entre Disciplinas - {dataset_name}',
                    xaxis_title='Média de Notas',
                    yaxis_title='Escola',
                    barmode='group',  # Barras agrupadas
                    height=600,
                    showlegend=True
                )
                
                st.plotly_chart(fig_comparacao, use_container_width=True)
            else:
                st.info("Selecione mais de uma disciplina para ver a comparação entre elas.")
        
        with tab3:
            # Box plot da distribuição por disciplina
            fig_box = px.box(
                dados_melted,
                x='Nota',
                y='Disciplina',
                color='Disciplina',
                orientation='h',  # Horizontal
                title=f'🎯 Distribuição de Notas por Disciplina - {dataset_name}',
                points="all"
            )
            
            fig_box.update_layout(
                yaxis_title='Disciplina',
                xaxis_title='Nota',
                height=500,
                showlegend=False
            )
            
            st.plotly_chart(fig_box, use_container_width=True)
            
            # Box plot por escola (apenas para top escolas) - ORDENADO MAIOR → MENOR
            st.subheader("📦 Distribuição por Escola (Top 15)")
            top_escolas = data_clean[coluna_agrupamento].value_counts().head(15).index
            data_top_escolas = data_clean[data_clean[coluna_agrupamento].isin(top_escolas)]
            
            if len(data_top_escolas) > 0:
                dados_top_melted = data_top_escolas.melt(
                    id_vars=[coluna_agrupamento],
                    value_vars=colunas_analise,
                    var_name='Disciplina',
                    value_name='Nota'
                )
                
                # Ordenar escolas pela média (MAIOR → MENOR)
                ordem_top_escolas = dados_top_melted.groupby(coluna_agrupamento)['Nota'].mean().sort_values(ascending=False).index
                
                fig_box_escolas = px.box(
                    dados_top_melted,
                    x='Nota',
                    y=coluna_agrupamento,
                    color='Disciplina',
                    orientation='h',
                    title=f'Distribuição de Notas - Top 15 Escolas - {dataset_name}',
                    category_orders={coluna_agrupamento: ordem_top_escolas}
                )
                
                fig_box_escolas.update_layout(
                    yaxis_title='Escola',
                    xaxis_title='Nota',
                    height=600
                )
                
                st.plotly_chart(fig_box_escolas, use_container_width=True)
        
        with tab4:
            # Estatísticas resumidas por disciplina
            st.subheader("📋 Estatísticas Detalhadas por Disciplina")
            
            for disciplina in colunas_analise:
                with st.expander(f"📚 {disciplina}", expanded=False):
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Média Geral", f"{data_clean[disciplina].mean():.2f}")
                        st.metric("Mediana", f"{data_clean[disciplina].median():.2f}")
                    
                    with col2:
                        st.metric("Desvio Padrão", f"{data_clean[disciplina].std():.2f}")
                        st.metric("Variância", f"{data_clean[disciplina].var():.2f}")
                    
                    with col3:
                        st.metric("Mínimo", f"{data_clean[disciplina].min():.2f}")
                        st.metric("Máximo", f"{data_clean[disciplina].max():.2f}")
                    
                    with col4:
                        st.metric("Assimetria", f"{data_clean[disciplina].skew():.2f}")
                        st.metric("Curtose", f"{data_clean[disciplina].kurtosis():.2f}")
                    
                    # Top 5 escolas nessa disciplina (MAIOR → MENOR)
                    st.subheader(f"🏅 Top 5 Escolas - {disciplina}")
                    top_escolas_disciplina = data_clean.groupby(coluna_agrupamento)[disciplina].mean().nlargest(5).round(2)
                    for i, (escola, media) in enumerate(top_escolas_disciplina.items(), 1):
                        medal = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][i-1]
                        st.write(f"{medal} **{escola}:** {media}")
    
    else:
        # Análise genérica para outras combinações de colunas (mantida igual)
        st.subheader("📈 Análise Estatística Geral")
        
        # Preparar dados
        colunas_analise_selecionadas = [coluna_agrupamento] + colunas_analise
        data_analysis = data[colunas_analise_selecionadas].copy()
        
        # Converter colunas numéricas
        for col in colunas_analise:
            if data_analysis[col].dtype == 'object':
                data_analysis[col] = pd.to_numeric(data_analysis[col], errors='coerce')
        
        # Remover valores nulos
        data_clean = data_analysis.dropna()
        
        if len(data_clean) == 0:
            st.error("❌ Não há dados válidos para análise após a limpeza.")
            return
        
        # Estatísticas agrupadas
        estatisticas_agrupadas = data_clean.groupby(coluna_agrupamento)[colunas_analise].agg(['mean', 'std', 'count']).round(2)
        estatisticas_agrupadas.columns = ['_'.join(col).strip() for col in estatisticas_agrupadas.columns.values]
        estatisticas_agrupadas = estatisticas_agrupadas.reset_index()
        
        st.subheader("📊 Estatísticas Agrupadas")
        st.dataframe(estatisticas_agrupadas, use_container_width=True)
        
        # Gráficos para análise genérica também em formato hierárquico
        if len(colunas_analise) > 0:
            st.subheader("📈 Visualizações")
            
            # Preparar dados para gráficos
            dados_melted = data_clean.melt(
                id_vars=[coluna_agrupamento],
                value_vars=colunas_analise,
                var_name='Variável',
                value_name='Valor'
            )
            
            tab1, tab2 = st.tabs(["📊 Ranking Hierárquico", "🎯 Distribuição"])
            
            with tab1:
                # Gráfico hierárquico para análise genérica
                medias_agrupadas = dados_melted.groupby([coluna_agrupamento, 'Variável'])['Valor'].mean().reset_index()
                
                # Ordenar pela média geral (MAIOR → MENOR)
                ordem_categorias = medias_agrupadas.groupby(coluna_agrupamento)['Valor'].mean().sort_values(ascending=False).index
                
                fig_hierarquico = px.bar(
                    medias_agrupadas,
                    x='Valor',
                    y=coluna_agrupamento,
                    color='Variável',
                    orientation='h',
                    title=f'Ranking por {coluna_agrupamento} - {dataset_name}',
                    hover_data={'Valor': ':.2f'},
                    category_orders={coluna_agrupamento: ordem_categorias}
                )
                
                fig_hierarquico.update_layout(
                    yaxis_title=coluna_agrupamento,
                    xaxis_title='Valor Médio',
                    height=600
                )
                
                st.plotly_chart(fig_hierarquico, use_container_width=True)
            
            with tab2:
                # Box plot da distribuição (horizontal)
                fig_box = px.box(
                    dados_melted,
                    x='Valor',
                    y='Variável',
                    color='Variável',
                    orientation='h',
                    title=f'Distribuição por Variável - {dataset_name}',
                    points="all"
                )
                
                fig_box.update_layout(
                    yaxis_title='Variável',
                    xaxis_title='Valor',
                    height=500
                )
                
                st.plotly_chart(fig_box, use_container_width=True)
    
    # Download dos dados processados
    st.subheader("💾 Exportar Resultados")
    
    if 'estatisticas_escolas' in locals() or 'estatisticas_agrupadas' in locals():
        dados_exportar = estatisticas_escolas if 'estatisticas_escolas' in locals() else estatisticas_agrupadas
        csv = dados_exportar.to_csv(index=False)
        
        st.download_button(
            label="📥 Baixar estatísticas em CSV",
            data=csv,
            file_name=f"analise_colunas_{dataset_name.lower().replace(' ', '_')}.csv",
            mime="text/csv"
        )


# Função para análise de correlação
def correlation_analysis(data, dataset_name):
    st.header(f"🔗 Análise de Correlação: Médias vs Quantidade de Alunos - {dataset_name}")
    
    # Encontrar colunas automaticamente
    st.subheader("🎯 Identificação das Colunas")
    
    # Procurar por colunas similares
    escola_keywords = ['escola', 'school', 'colégio', 'colegio', 'instituição', 'unidade', 'qual a sua escola']
    total_pontos_keywords = ['total', 'pontos', 'points', 'score', 'nota total', 'total de pontos']
    nota_lp_keywords = ['nota lp', 'nota_lp', 'portugues', 'português', 'lingua portuguesa', 'lp', 'língua portuguesa']
    nota_mat_keywords = ['nota mat', 'nota_mat', 'matemática', 'matematica', 'math', 'mat']
    
    # Usar função find_similar_columns corretamente
    colunas_escola_encontradas = find_similar_columns(data, escola_keywords)
    colunas_total_pontos_encontradas = find_similar_columns(data, total_pontos_keywords)
    colunas_nota_lp_encontradas = find_similar_columns(data, nota_lp_keywords)
    colunas_nota_mat_encontradas = find_similar_columns(data, nota_mat_keywords)
    
    # Mostrar colunas encontradas
    st.write("**Colunas identificadas:**")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Escola:** {colunas_escola_encontradas[0] if colunas_escola_encontradas else 'Não encontrada'}")
        st.write(f"**Total de Pontos:** {colunas_total_pontos_encontradas[0] if colunas_total_pontos_encontradas else 'Não encontrada'}")
    
    with col2:
        st.write(f"**Nota LP:** {colunas_nota_lp_encontradas[0] if colunas_nota_lp_encontradas else 'Não encontrada'}")
        st.write(f"**Nota MAT:** {colunas_nota_mat_encontradas[0] if colunas_nota_mat_encontradas else 'Não encontrada'}")
    
    # Verificar se encontrou a coluna de escola
    if not colunas_escola_encontradas:
        st.error("❌ Coluna 'QUAL A SUA ESCOLA?' não encontrada!")
        st.info("Selecione manualmente a coluna correspondente:")
        coluna_escola = st.selectbox(
            f"Selecione a coluna de Escola - {dataset_name}",
            data.columns,
            key=f"corr_escola_{dataset_name}"
        )
    else:
        coluna_escola = colunas_escola_encontradas[0]
    
    # Verificar e selecionar colunas de desempenho
    colunas_desempenho = []
    
    if colunas_total_pontos_encontradas:
        coluna_total_pontos = colunas_total_pontos_encontradas[0]
        colunas_desempenho.append(coluna_total_pontos)
    else:
        st.warning("⚠️ Coluna 'Total de pontos' não encontrada automaticamente.")
        coluna_total_pontos = st.selectbox(
            f"Selecione a coluna de Total de Pontos - {dataset_name}",
            data.columns,
            key=f"corr_total_{dataset_name}"
        )
        if coluna_total_pontos:
            colunas_desempenho.append(coluna_total_pontos)
    
    if colunas_nota_lp_encontradas:
        coluna_nota_lp = colunas_nota_lp_encontradas[0]
        colunas_desempenho.append(coluna_nota_lp)
    else:
        st.warning("⚠️ Coluna 'NOTA LP' não encontrada automaticamente.")
        coluna_nota_lp = st.selectbox(
            f"Selecione a coluna de Nota LP - {dataset_name}",
            data.columns,
            key=f"corr_lp_{dataset_name}"
        )
        if coluna_nota_lp:
            colunas_desempenho.append(coluna_nota_lp)
    
    if colunas_nota_mat_encontradas:
        coluna_nota_mat = colunas_nota_mat_encontradas[0]
        colunas_desempenho.append(coluna_nota_mat)
    else:
        st.warning("⚠️ Coluna 'NOTA MAT' não encontrada automaticamente.")
        coluna_nota_mat = st.selectbox(
            f"Selecione a coluna de Nota MAT - {dataset_name}",
            data.columns,
            key=f"corr_mat_{dataset_name}"
        )
        if coluna_nota_mat:
            colunas_desempenho.append(coluna_nota_mat)
    
    if not colunas_desempenho:
        st.error("❌ Nenhuma coluna de desempenho foi selecionada.")
        return
    
    # Preparar dados para análise
    st.subheader("🔄 Calculando Médias e Quantidade de Alunos por Escola")
    
    # Selecionar colunas relevantes
    colunas_analise = [coluna_escola] + colunas_desempenho
    
    st.write(f"**Colunas selecionadas para análise:**")
    st.write(f"- Agrupamento: {coluna_escola}")
    st.write(f"- Métricas: {', '.join(colunas_desempenho)}")
    
    # Criar cópia dos dados
    data_analysis = data[colunas_analise].copy()
    
    # Mostrar amostra dos dados
    with st.expander("📊 Ver amostra dos dados selecionados"):
        st.dataframe(data_analysis.head(10))
    
    # Converter colunas de desempenho para numérico
    st.write("**Convertendo colunas para formato numérico...**")
    for coluna in colunas_desempenho:
        if data_analysis[coluna].dtype == 'object':
            data_analysis[coluna] = pd.to_numeric(data_analysis[coluna], errors='coerce')
            st.write(f"- {coluna}: convertida para numérico")
        else:
            st.write(f"- {coluna}: já é numérico")
    
    # Remover valores nulos
    initial_count = len(data_analysis)
    data_clean = data_analysis.dropna()
    final_count = len(data_clean)
    
    st.write(f"**Limpeza de dados:** {final_count} registros válidos de {initial_count} total")
    
    if final_count == 0:
        st.error("❌ Não há dados válidos para análise após a limpeza.")
        return
    
    # Calcular estatísticas por escola
    st.write("**📈 Calculando médias e quantidade de alunos por escola...**")
    
    # Criar dicionário de agregação
    agg_dict = {}
    for coluna in colunas_desempenho:
        agg_dict[coluna] = 'mean'  # Calcular média para cada coluna de desempenho
    
    # Adicionar contagem de alunos (quantidade)
    agg_dict[colunas_desempenho[0]] = ['count', 'mean']  # Count para quantidade, mean para média
    
    # Agrupar por escola
    medias_por_escola = data_clean.groupby(coluna_escola).agg(agg_dict)
    
    # Corrigir nomes das colunas
    medias_por_escola.columns = ['_'.join(col).strip() for col in medias_por_escola.columns.values]
    medias_por_escola = medias_por_escola.reset_index()
    
    # Renomear colunas para nomes mais claros
    rename_dict = {coluna_escola: 'Escola'}
    
    # Encontrar e renomear a coluna de quantidade (count)
    for col in medias_por_escola.columns:
        if '_count' in col:
            rename_dict[col] = 'Quantidade_Alunos'
            break
    
    # Renomear colunas de média
    for coluna in colunas_desempenho:
        for col in medias_por_escola.columns:
            if f'{coluna}_mean' in col:
                rename_dict[col] = f'Media_{coluna}'
    
    medias_por_escola = medias_por_escola.rename(columns=rename_dict)
    
    # Garantir que temos a coluna Quantidade_Alunos
    if 'Quantidade_Alunos' not in medias_por_escola.columns:
        contagem_por_escola = data_clean.groupby(coluna_escola).size().reset_index(name='Quantidade_Alunos')
        medias_por_escola = medias_por_escola.merge(contagem_por_escola, on='Escola')
    
    # Classificar escolas por tamanho
    medias_por_escola['Tamanho_Escola'] = pd.cut(
        medias_por_escola['Quantidade_Alunos'],
        bins=[0, 50, 100, 200, float('inf')],
        labels=['Pequena (≤50)', 'Média (51-100)', 'Grande (101-200)', 'Muito Grande (>200)']
    )
    
    st.success(f"✅ Calculadas estatísticas para {len(medias_por_escola)} escolas")
    
    # Mostrar resumo
    st.subheader("📊 Resumo das Estatísticas por Escola")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total de Escolas", len(medias_por_escola))
        st.metric("Total de Alunos", medias_por_escola['Quantidade_Alunos'].sum())
    
    with col2:
        media_alunos = medias_por_escola['Quantidade_Alunos'].mean()
        st.metric("Média de Alunos por Escola", f"{media_alunos:.1f}")
        
        if 'Media_Total_Pontos' in medias_por_escola.columns:
            media_pontos = medias_por_escola['Media_Total_Pontos'].mean()
            st.metric("Média Geral - Total Pontos", f"{media_pontos:.2f}")
    
    with col3:
        if 'Media_Nota_LP' in medias_por_escola.columns:
            media_lp = medias_por_escola['Media_Nota_LP'].mean()
            st.metric("Média Geral - Nota LP", f"{media_lp:.2f}")
        
        if 'Media_Nota_MAT' in medias_por_escola.columns:
            media_mat = medias_por_escola['Media_Nota_MAT'].mean()
            st.metric("Média Geral - Nota MAT", f"{media_mat:.2f}")
    
    # Distribuição por tamanho de escola
    st.subheader("🏫 Distribuição das Escolas por Tamanho")
    
    dist_tamanho = medias_por_escola['Tamanho_Escola'].value_counts().sort_index()
    
    fig_dist = px.pie(
        values=dist_tamanho.values,
        names=dist_tamanho.index,
        title='Distribuição das Escolas por Tamanho',
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig_dist.update_layout(height=400)
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with col2:
        st.write("**Resumo por Tamanho:**")
        for tamanho, count in dist_tamanho.items():
            st.write(f"**{tamanho}:** {count} escolas")

    # ANÁLISE DE CORRELAÇÃO COM DUAS VISUALIZAÇÕES
    st.subheader("📈 Análise de Correlação: Duas Perspectivas")
    
    # Selecionar colunas numéricas para correlação
    colunas_media = [col for col in medias_por_escola.columns if col.startswith('Media_')]
    
    if not colunas_media:
        st.error("❌ Não há colunas de média calculadas para análise de correlação.")
        return
    
    # Calcular matriz de correlação
    colunas_correlacao = ['Quantidade_Alunos'] + colunas_media
    data_corr = medias_por_escola[colunas_correlacao]
    corr_matrix = data_corr.corr().round(3)
    
    # Focar nas correlações com Quantidade_Alunos
    correlacoes_com_quantidade = corr_matrix.loc['Quantidade_Alunos'].drop('Quantidade_Alunos')
    
    # Gráficos de correlação DUPLOS para cada métrica
    st.write("### 📊 Análise Gráfica Dupla por Métrica")
    
    for coluna_media in colunas_media:
        nome_metrica = coluna_media.replace('Media_', '').replace('_', ' ').title()
        correlacao = correlacoes_com_quantidade[coluna_media]
        
        st.write(f"#### 📈 {nome_metrica}")
        
        # Criar duas colunas para os gráficos
        col1, col2 = st.columns(2)
        
        with col1:
            # GRÁFICO 1: Com diferenciação por tamanho
            fig_tamanho = px.scatter(
                medias_por_escola,
                x='Quantidade_Alunos',
                y=coluna_media,
                size='Quantidade_Alunos',
                color='Tamanho_Escola',
                size_max=25,
                title=f'<b>Por Tamanho da Escola</b><br><sub>Correlação: r = {correlacao:.3f}</sub>',
                trendline="ols",
                hover_data=['Escola'],
                hover_name='Escola',
                labels={
                    'Quantidade_Alunos': 'Quantidade de Alunos',
                    coluna_media: f'Média de {nome_metrica}',
                    'Tamanho_Escola': 'Tamanho da Escola',
                    'size': 'Quantidade de Alunos'
                },
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            
            fig_tamanho.update_layout(
                height=500,
                showlegend=True,
                xaxis_title="Quantidade de Alunos",
                yaxis_title=f"Média de {nome_metrica}",
                font=dict(size=11)
            )
            
            fig_tamanho.update_traces(
                marker=dict(opacity=0.7, line=dict(width=1, color='darkgray')),
                line=dict(dash='solid', width=3, color='red')
            )
            
            st.plotly_chart(fig_tamanho, use_container_width=True)
        
        with col2:
            # GRÁFICO 2: Todos os pontos uniformes (análise limpa)
            fig_uniforme = px.scatter(
                medias_por_escola,
                x='Quantidade_Alunos',
                y=coluna_media,
                title=f'<b>Visão Geral</b><br><sub>Correlação: r = {correlacao:.3f}</sub>',
                trendline="ols",
                hover_data=['Escola', 'Tamanho_Escola'],
                hover_name='Escola',
                labels={
                    'Quantidade_Alunos': 'Quantidade de Alunos',
                    coluna_media: f'Média de {nome_metrica}'
                },
                color_discrete_sequence=['#1f77b4']  # Azul padrão
            )
            
            fig_uniforme.update_layout(
                height=500,
                showlegend=False,
                xaxis_title="Quantidade de Alunos",
                yaxis_title=f"Média de {nome_metrica}",
                font=dict(size=11)
            )
            
            fig_uniforme.update_traces(
                marker=dict(
                    size=8, 
                    opacity=0.7, 
                    line=dict(width=1, color='darkgray')
                ),
                line=dict(dash='solid', width=3, color='red')
            )
            
            st.plotly_chart(fig_uniforme, use_container_width=True)
        
        # Análise detalhada por tamanho de escola (abaixo dos gráficos)
        st.write(f"**📋 Análise Detalhada de {nome_metrica} por Tamanho de Escola:**")
        
        # Calcular estatísticas por tamanho
        stats_por_tamanho = medias_por_escola.groupby('Tamanho_Escola').agg({
            coluna_media: ['mean', 'std', 'count'],
            'Quantidade_Alunos': 'mean'
        }).round(2)
        
        # Formatando a tabela
        stats_por_tamanho.columns = ['Média', 'Desvio Padrão', 'Nº Escolas', 'Média Alunos']
        stats_por_tamanho = stats_por_tamanho.reset_index()
        
        # Mostrar tabela e métricas lado a lado
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.dataframe(stats_por_tamanho, use_container_width=True)
        
        with col2:
            st.metric(
                "Correlação Geral",
                f"{correlacao:.3f}",
                interpretar_correlacao_simples(correlacao)
            )
            
            # Encontrar melhor desempenho por tamanho
            if not stats_por_tamanho.empty:
                melhor_tamanho = stats_por_tamanho.loc[stats_por_tamanho['Média'].idxmax()]
                st.metric(
                    "Melhor Desempenho",
                    f"{melhor_tamanho['Tamanho_Escola']}",
                    f"Média: {melhor_tamanho['Média']}"
                )
        
        st.write("---")
    
    # Tabela resumo de correlações
    st.subheader("📋 Resumo Geral das Correlações")
    
    resumo_correlacoes = []
    for coluna_media in colunas_media:
        nome_metrica = coluna_media.replace('Media_', '').replace('_', ' ').title()
        correlacao = correlacoes_com_quantidade[coluna_media]
        
        resumo_correlacoes.append({
            'Métrica': nome_metrica,
            'Correlação (r)': correlacao,
            'Interpretação': interpretar_correlacao_simples(correlacao),
            'Força': classificar_forca_correlacao(correlacao)
        })
    
    resumo_df = pd.DataFrame(resumo_correlacoes)
    st.dataframe(resumo_df, use_container_width=True)
    
    # Análise comparativa final
    st.subheader("📈 Visão Comparativa das Correlações")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Gráfico de barras comparativo
        fig_comparativo = px.bar(
            resumo_df,
            x='Métrica',
            y='Correlação (r)',
            color='Correlação (r)',
            color_continuous_scale='RdBu_r',
            title='Comparação das Correlações',
            text='Correlação (r)',
            hover_data=['Interpretação']
        )
        
        fig_comparativo.update_layout(
            xaxis_title='Métrica de Desempenho',
            yaxis_title='Coeficiente de Correlação (r)',
            height=400
        )
        
        fig_comparativo.update_traces(
            texttemplate='%{text:.3f}', 
            textposition='outside'
        )
        
        st.plotly_chart(fig_comparativo, use_container_width=True)
    
    with col2:
        # Resumo estatístico
        st.write("**📊 Resumo Estatístico**")
        
        if not resumo_df.empty:
            # Análise geral
            correlacoes_positivas = len([r for r in resumo_df['Correlação (r)'] if r > 0])
            correlacoes_negativas = len([r for r in resumo_df['Correlação (r)'] if r < 0])
            
            st.metric("Correlações Positivas", correlacoes_positivas)
            st.metric("Correlações Negativas", correlacoes_negativas)
            
            maior_corr = resumo_df.loc[resumo_df['Correlação (r)'].abs().idxmax()]
            st.metric(
                "Maior Correlação (abs)", 
                f"{maior_corr['Correlação (r)']:.3f}",
                f"{maior_corr['Métrica']}"
            )
            
            # Distribuição por força
            st.write("**Força das Correlações:**")
            for forca in ['Forte', 'Moderada', 'Fraca', 'Muito Fraca']:
                count = len(resumo_df[resumo_df['Força'] == forca])
                if count > 0:
                    st.write(f"- {forca}: {count}")
    
    # Insights finais
    st.subheader("💡 Insights e Recomendações")
    
    if not resumo_df.empty:
        # Análise geral
        correlacoes_positivas = len([r for r in resumo_df['Correlação (r)'] if r > 0])
        correlacoes_negativas = len([r for r in resumo_df['Correlação (r)'] if r < 0])
        
        st.write(f"**📈 Distribuição das Correlações:**")
        st.write(f"- **{correlacoes_positivas} correlação(ões) positiva(s)** - Escolas maiores tendem a ter melhor desempenho")
        st.write(f"- **{correlacoes_negativas} correlação(ões) negativa(s)** - Escolas menores tendem a ter melhor desempenho")
        
        # Recomendações baseadas nos resultados
        st.write("**🎯 Recomendações Estratégicas:**")
        
        if any(r > 0.5 for r in resumo_df['Correlação (r)']):
            st.success("""
            **Forte evidência de vantagem das escolas maiores:**
            - Considere políticas que aproveitem economias de escala
            - Investir em infraestrutura para escolas maiores
            - Desenvolver programas específicos para escolas de grande porte
            """)
        elif any(r < -0.5 for r in resumo_df['Correlação (r)']):
            st.warning("""
            **Forte evidência de vantagem das escolas menores:**
            - Avaliar estratégias de descentralização
            - Considerar criação de unidades menores
            - Focar em personalização do ensino
            """)
        else:
            st.info("""
            **Relação limitada entre tamanho e desempenho:**
            - Focar em outros fatores determinantes
            - Qualidade docente e formação continuada
            - Infraestrutura adequada
            - Gestão escolar eficiente
            - Participação da comunidade
            """)
        
        # Destaques por categoria de tamanho
        st.write("**🏫 Destaques por Categoria de Tamanho:**")
        
        # Encontrar a categoria com melhor desempenho médio geral
        desempenho_por_tamanho = {}
        for coluna_media in colunas_media:
            stats = medias_por_escola.groupby('Tamanho_Escola')[coluna_media].mean()
            for tamanho, media in stats.items():
                if tamanho not in desempenho_por_tamanho:
                    desempenho_por_tamanho[tamanho] = []
                desempenho_por_tamanho[tamanho].append(media)
        
        # Calcular média geral por tamanho
        media_geral_por_tamanho = {
            tamanho: sum(medias) / len(medias) 
            for tamanho, medias in desempenho_por_tamanho.items()
        }
        
        if media_geral_por_tamanho:
            melhor_tamanho_geral = max(media_geral_por_tamanho, key=media_geral_por_tamanho.get)
            st.write(f"- **Melhor desempenho médio:** {melhor_tamanho_geral}")
            st.write(f"- **Média geral:** {media_geral_por_tamanho[melhor_tamanho_geral]:.2f}")

# Funções auxiliares (manter as mesmas)
def interpretar_correlacao_detalhada(r):
    """Interpretação detalhada do coeficiente de correlação"""
    if r >= 0.7:
        return "Correlação positiva forte: escolas com mais alunos tendem a ter desempenho significativamente melhor"
    elif r >= 0.5:
        return "Correlação positiva moderada: existe uma relação positiva entre tamanho da escola e desempenho"
    elif r >= 0.3:
        return "Correlação positiva fraca: leve tendência de escolas maiores terem melhor desempenho"
    elif r > -0.3:
        return "Correlação muito fraca ou nula: praticamente não há relação entre tamanho da escola e desempenho"
    elif r > -0.5:
        return "Correlação negativa fraca: leve tendência de escolas menores terem melhor desempenho"
    elif r > -0.7:
        return "Correlação negativa moderada: escolas menores tendem a ter melhor desempenho"
    else:
        return "Correlação negativa forte: escolas com menos alunos tendem a ter desempenho significativamente melhor"

def interpretar_correlacao_simples(r):
    """Interpretação simples do coeficiente de correlação"""
    if r >= 0.7:
        return "Correlação positiva forte"
    elif r >= 0.5:
        return "Correlação positiva moderada"
    elif r >= 0.3:
        return "Correlação positiva fraca"
    elif r > -0.3:
        return "Correlação muito fraca ou nula"
    elif r > -0.5:
        return "Correlação negativa fraca"
    elif r > -0.7:
        return "Correlação negativa moderada"
    else:
        return "Correlação negativa forte"

def classificar_forca_correlacao(r):
    """Classifica a força da correlação"""
    abs_r = abs(r)
    if abs_r >= 0.7:
        return "Forte"
    elif abs_r >= 0.5:
        return "Moderada"
    elif abs_r >= 0.3:
        return "Fraca"
    else:
        return "Muito Fraca"



# Sidebar para navegação
st.sidebar.title("Esquenta SAEGO")
analysis_type = st.sidebar.radio(
    "Selecione o tipo de análise:",
    ["Informações Básicas", "Estatísticas por Escola", "Análise por Coluna", "Correlação", "Valores Ausentes", "Relatório Completo"]
)

# Carregar dados
dados_3anos = load_data("dados_3anos.csv")
dados_9anos = load_data("dados_9anos.csv")

# Verificar se os dados foram carregados
if dados_3anos is not None and dados_9anos is not None:
    
    # Criar abas para cada dataset
    tab1, tab2 = st.tabs(["📊 Dados 3° Séries", "📈 Dados 9° Anos"])
    
    with tab1:
        if analysis_type == "Informações Básicas":
            basic_statistics(dados_3anos, "Dados 3 Série")
        elif analysis_type == "Estatísticas por Escola":
            school_statistics(dados_3anos, "Dados 3 Série")
        elif analysis_type == "Análise por Coluna":
            column_analysis(dados_3anos, "Dados 3 Série")
        elif analysis_type == "Correlação":
            correlation_analysis(dados_3anos, "Dados 3 Série")
        elif analysis_type == "Valores Ausentes":
            missing_values_analysis(dados_3anos, "Dados 3 Série")
        elif analysis_type == "Relatório Completo":
            export_report(dados_3anos, "Dados 3 Série")
    
    with tab2:
        if analysis_type == "Informações Básicas":
            basic_statistics(dados_9anos, "Dados 9 Anos")
        elif analysis_type == "Estatísticas por Escola":
            school_statistics(dados_9anos, "Dados 9 Anos")
        elif analysis_type == "Análise por Coluna":
            column_analysis(dados_9anos, "Dados 9 Anos")
        elif analysis_type == "Correlação":
            correlation_analysis(dados_9anos, "Dados 9 Anos")
        elif analysis_type == "Valores Ausentes":
            missing_values_analysis(dados_9anos, "Dados 9 Anos")
        elif analysis_type == "Relatório Completo":
            export_report(dados_9anos, "Dados 9 Anos")
    
    # Comparação entre datasets (opcional)
    st.sidebar.markdown("---")
    if st.sidebar.checkbox("Mostrar Comparação entre Datasets"):
        st.header("🔄 Comparação entre Dados 3 Anos vs 9 Anos")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dados 3° Série")
            st.write(f"Registros: {dados_3anos.shape[0]}")
            st.write(f"Variáveis: {dados_3anos.shape[1]}")
            st.write(f"Valores nulos: {dados_3anos.isnull().sum().sum()}")
        
        with col2:
            st.subheader("Dados 9° Anos")
            st.write(f"Registros: {dados_9anos.shape[0]}")
            st.write(f"Variáveis: {dados_9anos.shape[1]}")
            st.write(f"Valores nulos: {dados_9anos.isnull().sum().sum()}")

else:
    st.error("Não foi possível carregar um ou ambos os arquivos de dados.")
    st.info("""
    Certifique-se de que:
    1. Os arquivos 'dados_3anos.csv' e 'dados_9anos.csv' estão no mesmo diretório deste script
    2. Os arquivos estão no formato CSV válido
    3. Os nomes dos arquivos estão corretos
    """)

# Rodapé
st.sidebar.markdown("---")
st.sidebar.info(
    "Desenvolvido com Streamlit | "
    "Esquenta SAEGO 2025 ÁGUAS LINDAS - GO"
)
